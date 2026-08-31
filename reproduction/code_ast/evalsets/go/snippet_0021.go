package main

// Counting service
//
// Using hyperloglog algorithm to count unique users.
//

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/tkawachi/countserv"
)

func getPort() string {
	const DefaultPort = "8080"
	if p, ok := os.LookupEnv("PORT"); ok {
		return p
	}
	return DefaultPort
}

const counterFile = "counter.json"

func saveCounter(counter *countserv.Counter) error {
	f, err := os.Create(counterFile)
	if err != nil {
		return err
	}
	defer f.Close()
	json.NewEncoder(f).Encode(counter)
	buf := bytes.NewBuffer(nil)
	json.NewEncoder(buf).Encode(counter)
	log.Println(buf.String())
	return nil
}

func loadOrNewCounter() (*countserv.Counter, error) {
	if _, err := os.Stat(counterFile); err != nil {
		return countserv.NewCounter(), nil
	}
	f, err := os.Open(counterFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var c countserv.Counter
	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return nil, err
	}
	return &c, nil
}

func main() {
	counter, err := loadOrNewCounter()
	if err != nil {
		log.Fatal(err)
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		item := r.URL.Query().Get("item")
		user := r.URL.Query().Get("user")
		if item == "" || user == "" {
			http.Error(w, "item and user are required", http.StatusBadRequest)
			return
		}
		changed := counter.Insert(item, user)
		if changed {
			err := saveCounter(counter)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		fmt.Fprintf(w, "OK")
	})

	http.HandleFunc("/estimates", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "%v", counter.Estimates())
	})

	port := getPort()
	log.Println("Listening on port:", port)
	srv := http.Server{Addr: ":" + port}
	ln, err := net.Listen("tcp", srv.Addr)
	if err != nil {
		log.Fatal(err)
	}
	go func() {
		if err := srv.Serve(ln); err != nil {
			log.Print(err)
		}
	}()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGTERM)
	<-sigCh
	log.Println("Shutting down...")
	if err := srv.Shutdown(context.Background()); err != nil {
		log.Fatal(err)
	}
	log.Println("Server gracefully stopped")
}