package gohost

import (
	"net/http"

	"github.com/eleniums/async"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

const (
	// DefaultGRPCAddr is the default address for the gRPC endpoint.
	DefaultGRPCAddr = "127.0.0.1:50051"

	// DefaultHTTPAddr is the default address for the HTTP endpoint.
	DefaultHTTPAddr = "127.0.0.1:9090"

	// DefaultDebugAddr is the default address for the debug endpoint (/debug/pprof and /debug/vars).
	DefaultDebugAddr = "127.0.0.1:6060"

	// DefaultMaxSendMsgSize is the default max send message size, per gRPC
	DefaultMaxSendMsgSize = 1024 * 1024 * 4

	// DefaultMaxRecvMsgSize is the default max receive message size, per gRPC
	DefaultMaxRecvMsgSize = 1024 * 1024 * 4
)

// GRPCServer is used to register a gRPC server.
type GRPCServer func(s *grpc.Server)

// HTTPGateway is used to register a HTTP gateway for forwarding requests to a gRPC endpoint.
type HTTPGateway func(ctx context.Context, mux *runtime.ServeMux, endpoint string, opts []grpc.DialOption) (err error)

// Hoster is used to serve gRPC and HTTP endpoints.
type Hoster struct {
	// GRPCAddr is the endpoint (host and port) on which to host the gRPC service. Default is 127.0.0.1:50051. May be left blank if no gRPC servers have been registered.
	GRPCAddr string

	// HTTPAddr is the endpoint (host and port) on which to host the HTTP service. Default is 127.0.0.1:9090. May be left blank if no HTTP gateways have been registered.
	HTTPAddr string

	// DebugAddr is the endpoint (host and port) on which to host the debug endpoint (/debug/pprof and /debug/vars). Default is 127.0.0.1:6060. May be left blank if EnableDebug is false.
	DebugAddr string

	// CertFile is the certificate file for use with TLS. May be left blank if using insecure mode.
	CertFile string

	// KeyFile is the private key file for use with TLS. May be left blank if using insecure mode.
	KeyFile string

	// InsecureSkipVerify will cause verification of the host name during a TLS handshake to be skipped if set to true.
	InsecureSkipVerify bool

	// HTTPHandler is used to register a handler that can optionally be added to the HTTP endpoint. Leave blank to use default mux.
	HTTPHandler func(mux *runtime.ServeMux) http.Handler

	// EnableDebug will enable the debug endpoint (/debug/pprof and /debug/vars). The debug endpoint address is defined by DebugAddr.
	EnableDebug bool

	// MaxSendMsgSize will change the size of the message that can be sent from the service.
	MaxSendMsgSize int

	// MaxRecvMsgSize will change the size of the message that can be received by the service.
	MaxRecvMsgSize int

	// UnaryInterceptors is an array of unary interceptors to be used by the service. They will be executed in order, from first to last.
	UnaryInterceptors []grpc.UnaryServerInterceptor

	// StreamInterceptors is an array of stream interceptors to be used by the service. They will be executed in order, from first to last.
	StreamInterceptors []grpc.StreamServerInterceptor

	// grpcServers is an array of gRPC servers to be hosted.
	grpcServers []GRPCServer

	// httpGateways is an array of HTTP gateways to be hosted.
	httpGateways []HTTPGateway
}

// NewHoster creates a new hoster instance with defaults set.
func NewHoster() *Hoster {
	return &Hoster{
		GRPCAddr:       DefaultGRPCAddr,
		HTTPAddr:       DefaultHTTPAddr,
		DebugAddr:      DefaultDebugAddr,
		MaxSendMsgSize: DefaultMaxSendMsgSize,
		MaxRecvMsgSize: DefaultMaxRecvMsgSize,
	}
}

// RegisterGRPCServer will add a function for registering a gRPC server. The function is invoked when ListenAndServe is called.
func (h *Hoster) RegisterGRPCServer(servers ...GRPCServer) {
	h.grpcServers = append(h.grpcServers, servers...)
}

// RegisterHTTPGateway will add a function for registering a HTTP gateway. The function is invoked when ListenAndServe is called.
func (h *Hoster) RegisterHTTPGateway(gateways ...HTTPGateway) {
	h.httpGateways = append(h.httpGateways, gateways...)
}

// ListenAndServe creates and starts the server.
func (h *Hoster) ListenAndServe() error {
	tasks := []async.Task{}

	// serve debug endpoint
	if h.EnableDebug {
		tasks = append(tasks, func() error {
			return h.serveDebug()
		})
	}

	// serve HTTP endpoint
	if len(h.httpGateways) > 0 {
		tasks = append(tasks, func() error {
			return h.serveHTTP()
		})
	}

	// serve gRPC endpoint
	if len(h.grpcServers) > 0 {
		tasks = append(tasks, func() error {
			return h.serveGRPC()
		})
	}

	errc := async.Run(tasks...)

	return async.Wait(errc)
}