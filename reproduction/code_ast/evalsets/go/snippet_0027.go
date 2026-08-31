package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	_ "github.com/joho/godotenv/autoload"

	"github.com/RisingStack/almandite-user-service/internal/config"
	"github.com/RisingStack/almandite-user-service/internal/dal"
	"github.com/RisingStack/almandite-user-service/internal/handlers"
	"github.com/RisingStack/almandite-user-service/internal/handlers/middleware"
	"github.com/RisingStack/almandite-user-service/internal/handlers/middleware/authentication"
)

const DefaultHTTPAddr = ":0"

var httpAddr string

func init() {
	flag.StringVar(&httpAddr, "addr", DefaultHTTPAddr, "Set the HTTP bind address")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n", os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()
	listener, err := net.Listen("tcp", httpAddr)
	if err != nil {
		log.Fatal(err)
	}

	tcpAddr := listener.Addr().(*net.TCPAddr)

	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, os.Interrupt, syscall.SIGTERM)

	db := dal.NewDAL()

	if err := db.OpenConnection(
		config.GetConfiguration().PostgresURL,
		config.GetConfiguration().DebugSQL,
	); err != nil {
		log.Fatal("Failed to open DB conenction", err)
	}

	go func() {
		select {
		case <-signalCh:
			log.Println("Signal received, shutting down...")
			db.CloseConnection()
			os.Exit(0)
		}
	}()

	log.Printf("Open the following URL in the browser: http://%s:%d\n", convertIPtoString(tcpAddr.IP), tcpAddr.Port)

	authStore := authentication.AuthStore{
		UserRepository: db.Users(),
	}

	http.HandleFunc("/healthcheck",
		middleware.Chain(
			middleware.Timer,
			middleware.Logger,
		)(handlers.Healthcheck))

	http.HandleFunc("/alma",
		middleware.Chain(
			middleware.Timer,
			middleware.Logger,
			authStore.Basic,
		)(
			func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprint(w, "korte")
			},
		),
	)

	if err := http.Serve(listener, nil); err != nil {
		log.Fatal(err)
	}
}

func convertIPtoString(ip net.IP) string {
	return fmt.Sprintf("%d.%d.%d.%d", ip[0], ip[1], ip[2], ip[3])
}