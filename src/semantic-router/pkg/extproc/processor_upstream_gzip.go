package extproc

import (
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var errGzipReplyTooLarge = errors.New("decoded gzip reply exceeds the upstream body limit")

// requestUpstreamGzip lets a buffered reply arrive compressed unless the client
// refused gzip. Streamed replies stay identity because the stream path parses
// SSE frames as they arrive. Respecting a refusal keeps a reply the router
// never sees, when the data plane withholds the body, readable by the client.
func requestUpstreamGzip(setHeaders *[]*core.HeaderValueOption, ctx *RequestContext) {
	if ctx == nil || ctx.ExpectStreamingResponse || !clientAcceptsGzip(ctx) {
		return
	}
	*setHeaders = append(*setHeaders, &core.HeaderValueOption{
		Header:       &core.HeaderValue{Key: "accept-encoding", RawValue: []byte("gzip")},
		AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
	})
	ctx.UpstreamGzipRequested = true
}

// clientAcceptsGzip treats a missing Accept-Encoding as accepting any coding,
// as RFC 9110 does. An empty value accepts none.
func clientAcceptsGzip(ctx *RequestContext) bool {
	for key, value := range ctx.Headers {
		if strings.EqualFold(key, "accept-encoding") {
			return acceptsGzip(value)
		}
	}
	return true
}

// acceptsGzip reports whether an Accept-Encoding value allows gzip. An explicit
// gzip entry takes precedence over a wildcard, and q=0 refuses.
func acceptsGzip(value string) bool {
	explicit, wildcard := -1.0, -1.0
	for _, entry := range strings.Split(value, ",") {
		name, params, _ := strings.Cut(entry, ";")
		switch strings.ToLower(strings.TrimSpace(name)) {
		case "gzip", "x-gzip":
			explicit = max(explicit, acceptQuality(params))
		case "*":
			wildcard = max(wildcard, acceptQuality(params))
		}
	}
	if explicit >= 0 {
		return explicit > 0
	}
	return wildcard > 0
}

// acceptQuality returns an entry's q value: 1 when absent, 0 when malformed.
func acceptQuality(params string) float64 {
	for _, param := range strings.Split(params, ";") {
		key, value, found := strings.Cut(param, "=")
		if !found || !strings.EqualFold(strings.TrimSpace(key), "q") {
			continue
		}
		quality, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err != nil {
			return 0
		}
		return quality
	}
	return 1
}

// markUpstreamGzipReply runs for every response, so a fallback reply that
// arrives uncompressed is not decoded on the strength of an earlier one.
func markUpstreamGzipReply(headerMap *core.HeaderMap, ctx *RequestContext) {
	if ctx == nil {
		return
	}
	ctx.UpstreamGzipReply = false
	if !ctx.UpstreamGzipRequested || headerMap == nil {
		return
	}
	for _, header := range headerMap.Headers {
		if strings.EqualFold(header.Key, "content-encoding") {
			encoding := strings.ToLower(strings.TrimSpace(extractHeaderValue(header)))
			ctx.UpstreamGzipReply = encoding == "gzip" || encoding == "x-gzip"
			return
		}
	}
}

func upstreamGzipBodyPending(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) bool {
	return ctx != nil && ctx.UpstreamGzipReply && !ctx.IsStreamingResponse && v.ResponseBody.GetEndOfStream()
}

// handleUpstreamGzipBody decodes the reply before any response phase reads it.
// The client receives decoded bytes, so the provider's content-encoding goes.
func (r *OpenAIRouter) handleUpstreamGzipBody(v *ext_proc.ProcessingRequest_ResponseBody, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	ctx.UpstreamGzipReply = false
	decoded, err := decodeGzipReply(v.ResponseBody.GetBody(), llmprotocol.DefaultPolicy().Limits.BodyBytes)
	if err != nil {
		logging.ComponentWarnEvent("extproc", "upstream_gzip_decode_failed", map[string]interface{}{
			"request_id": ctx.RequestID,
			"error":      err.Error(),
		})
		return r.createErrorResponse(502, "The selected model returned an invalid response"), nil
	}
	response, err := r.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: decoded, EndOfStream: true},
	}, ctx)
	if err != nil {
		return response, err
	}
	deliverDecodedReply(response, decoded)
	return response, nil
}

// decodeGzipReply passes an empty body through, so a bodyless error keeps the
// provider's status instead of becoming a decode failure.
func decodeGzipReply(body []byte, limit int) ([]byte, error) {
	if len(body) == 0 {
		return body, nil
	}
	reader, err := gzip.NewReader(bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	decoded, err := io.ReadAll(io.LimitReader(reader, int64(limit)+1))
	if err != nil {
		return nil, err
	}
	if len(decoded) > limit {
		return nil, errGzipReplyTooLarge
	}
	return decoded, nil
}

// deliverDecodedReply keeps a body another phase rewrote and otherwise sends
// the decoded bytes, without the provider's encoding or byte count.
func deliverDecodedReply(response *ext_proc.ProcessingResponse, decoded []byte) {
	bodyResponse, ok := response.GetResponse().(*ext_proc.ProcessingResponse_ResponseBody)
	if !ok {
		return
	}
	if bodyResponse.ResponseBody.GetResponse().GetBodyMutation() == nil {
		setResponseBodyMutation(response, decoded)
	}
	common := bodyResponse.ResponseBody.Response
	if common.HeaderMutation == nil {
		common.HeaderMutation = &ext_proc.HeaderMutation{}
	}
	ensureHeaderRemoved(common.HeaderMutation, "content-encoding")
	ensureHeaderRemoved(common.HeaderMutation, "content-length")
}
