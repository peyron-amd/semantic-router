package extproc

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"slices"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const astralReply = `{"id":"msg_1","type":"message","role":"assistant","model":"provider-model","content":[{"type":"text","text":"💡 done"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":4}}`

func TestAcceptsGzip(t *testing.T) {
	for value, want := range map[string]bool{
		"":                    false,
		"identity":            false,
		"br, deflate":         false,
		"gzip":                true,
		"GZIP":                true,
		"gzip, deflate":       true,
		"deflate, gzip;q=0.5": true,
		"x-gzip":              true,
		"gzip;q=0":            false,
		"gzip; q=0.0":         false,
		"gzip;q=bogus":        false,
		"*":                   true,
		"*;q=0":               false,
		"gzip;q=0, *":         false,
	} {
		if got := acceptsGzip(value); got != want {
			t.Errorf("acceptsGzip(%q) = %t, want %t", value, got, want)
		}
	}
}

func TestProviderDispatchAsksForGzipOnlyForABufferedReplyTheClientAccepts(t *testing.T) {
	for _, tc := range []struct {
		name           string
		acceptEncoding string
		streaming      bool
		want           string
	}{
		{name: "client accepts gzip", acceptEncoding: "gzip, deflate", want: "gzip"},
		{name: "client sends no Accept-Encoding", want: "gzip"},
		{name: "client asks for identity", acceptEncoding: "identity"},
		{name: "client refuses gzip", acceptEncoding: "gzip;q=0"},
		{name: "streamed reply", acceptEncoding: "gzip, deflate", streaming: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx, got := dispatchForGzipTest(t, llmprotocol.AnthropicMessagesV1, tc.acceptEncoding, tc.streaming)
			if got != tc.want || ctx.UpstreamGzipRequested != (tc.want != "") {
				t.Fatalf("upstream accept-encoding = %q, requested = %t; want %q", got, ctx.UpstreamGzipRequested, tc.want)
			}
		})
	}
}

func TestGzipReplyIsDecodedBeforeTheRouterReadsIt(t *testing.T) {
	t.Run("same wire format", func(t *testing.T) {
		router, ctx, _ := dispatchForGzipTest(t, llmprotocol.AnthropicMessagesV1, "gzip, deflate", false)
		common := sendGzipReply(t, router, ctx, gzipForTest(t, astralReply)).GetResponseBody().GetResponse()
		var reply struct{ Content []struct{ Text string } }
		if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &reply); err != nil ||
			len(reply.Content) != 1 || reply.Content[0].Text != "💡 done" {
			t.Fatalf("client body = %s (%v)", common.GetBodyMutation().GetBody(), err)
		}
		assertDecodedReplyHeaders(t, common)
	})
	t.Run("translated for a chat client", func(t *testing.T) {
		router, ctx, _ := dispatchForGzipTest(t, llmprotocol.OpenAIChatV1, "gzip, deflate", false)
		common := sendGzipReply(t, router, ctx, gzipForTest(t, astralReply)).GetResponseBody().GetResponse()
		var reply struct {
			Choices []struct{ Message struct{ Content string } }
		}
		if err := json.Unmarshal(common.GetBodyMutation().GetBody(), &reply); err != nil ||
			len(reply.Choices) != 1 || reply.Choices[0].Message.Content != "💡 done" {
			t.Fatalf("client body = %s (%v)", common.GetBodyMutation().GetBody(), err)
		}
		assertDecodedReplyHeaders(t, common)
	})
}

func TestEveryReplyIsCheckedForGzipOnItsOwn(t *testing.T) {
	gzipped := &core.HeaderMap{Headers: []*core.HeaderValue{{Key: "content-encoding", Value: "gzip"}}}
	ctx := &RequestContext{UpstreamGzipRequested: true}
	markUpstreamGzipReply(gzipped, ctx)
	if !ctx.UpstreamGzipReply {
		t.Fatal("a requested gzip reply was not marked for decoding")
	}
	markUpstreamGzipReply(&core.HeaderMap{}, ctx)
	if ctx.UpstreamGzipReply {
		t.Fatal("an uncompressed fallback reply kept the earlier reply's gzip mark")
	}
	ctx.UpstreamGzipRequested = false
	markUpstreamGzipReply(gzipped, ctx)
	if ctx.UpstreamGzipReply {
		t.Fatal("a gzip reply the router did not ask for was marked for decoding")
	}
}

func TestGzipReplyThatCannotBeDecodedIsABadGateway(t *testing.T) {
	router, ctx, _ := dispatchForGzipTest(t, llmprotocol.AnthropicMessagesV1, "gzip", false)
	response := sendGzipReply(t, router, ctx, []byte(astralReply))
	if code := response.GetImmediateResponse().GetStatus().GetCode(); code != typev3.StatusCode_BadGateway {
		t.Fatalf("status = %v, want 502", code)
	}
}

func TestDecodeGzipReplyRefusesAReplyOverTheLimit(t *testing.T) {
	if _, err := decodeGzipReply(gzipForTest(t, string(bytes.Repeat([]byte("a"), 1025))), 1024); !errors.Is(err, errGzipReplyTooLarge) {
		t.Fatalf("over the limit: error = %v", err)
	}
	if decoded, err := decodeGzipReply(gzipForTest(t, "abc"), 1024); err != nil || string(decoded) != "abc" {
		t.Fatalf("under the limit: %q, %v", decoded, err)
	}
	if decoded, err := decodeGzipReply(nil, 1024); err != nil || len(decoded) != 0 {
		t.Fatalf("empty body: %q, %v", decoded, err)
	}
}

func dispatchForGzipTest(
	t *testing.T,
	clientFormat llmprotocol.WireFormat,
	acceptEncoding string,
	streaming bool,
) (*OpenAIRouter, *RequestContext, string) {
	t.Helper()
	router, model := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	request := testNeutralRequest(model, "hello")
	ctx := routingTestContext(clientFormat, request)
	if acceptEncoding != "" {
		ctx.Headers["accept-encoding"] = acceptEncoding
	}
	ctx.ExpectStreamingResponse = streaming
	dispatch, err := router.prepareProviderDispatch(request, model, "", false, ctx)
	if err != nil {
		t.Fatalf("prepareProviderDispatch: %v", err)
	}
	response, err := router.finalizeProviderDispatchResponse(dispatch, router.buildProviderDispatchResponse(dispatch, ctx), ctx)
	if err != nil {
		t.Fatalf("finalizeProviderDispatchResponse: %v", err)
	}
	set := headerValuesByName(response.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
	return router, ctx, set["accept-encoding"]
}

func sendGzipReply(t *testing.T, router *OpenAIRouter, ctx *RequestContext, body []byte) *ext_proc.ProcessingResponse {
	t.Helper()
	if _, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":status", Value: "200"},
			{Key: "content-type", Value: "application/json"},
			{Key: "content-encoding", Value: "gzip"},
		}}},
	}, ctx); err != nil {
		t.Fatalf("handleResponseHeaders: %v", err)
	}
	response, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("handleResponseBody: %v", err)
	}
	return response
}

func assertDecodedReplyHeaders(t *testing.T, common *ext_proc.CommonResponse) {
	t.Helper()
	removed := common.GetHeaderMutation().GetRemoveHeaders()
	if !slices.Contains(removed, "content-encoding") || !slices.Contains(removed, "content-length") {
		t.Fatalf("decoded reply keeps the provider's framing headers; removed = %v", removed)
	}
}

func gzipForTest(t *testing.T, body string) []byte {
	t.Helper()
	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)
	if _, err := writer.Write([]byte(body)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}
