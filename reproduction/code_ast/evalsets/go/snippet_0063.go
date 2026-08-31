package powerdns

import (
	"crypto/tls"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// ClientConn required data for interaction with the powerdns api
type ClientConn struct {
	Client  http.Client
	Records RRsets
	Input   map[string]string
}

// NewClientConn validate the user input is as expected and add it to a ClientConn structure.
func NewClientConn() (*ClientConn, error) {

	var ttl int

	input, err := CheckUserInput()
	if err != nil {
		return nil, err
	}

	conn := ClientConn{
		Input: input,
	}

	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: false},
	}
	cli := http.Client{Transport: tr}

	if conn.Input["action"] == "add" {

		ttl, err = strconv.Atoi(conn.Input["ttl"])
		if err != nil {
			return nil, err
		}

		record := RRsets{
			Sets: []RRset{
				RRset{
					Name:       conn.Input["record"],
					Type:       "A",
					TTL:        ttl,
					Changetype: "REPLACE",
					Records: []Records{
						Records{
							Content:  conn.Input["ip"],
							Disabled: false,
						},
					},
					Comments: []Comments{
						Comments{
							Content:    "Automatically created via Http API",
							Account:    "Infra Services",
							ModifiedAT: int64(time.Now().Unix()),
						},
					},
				},
			},
		}

		conn.Client = cli
		conn.Records = record

	} else if conn.Input["action"] == "delete" {
		record := RRsets{
			Sets: []RRset{
				RRset{
					Name:       conn.Input["record"],
					Type:       "A",
					TTL:        ttl,
					Changetype: "DELETE",
					Records:    []Records{},
					Comments:   []Comments{},
				},
			},
		}

		conn.Client = cli
		conn.Records = record

	} else {
		return nil, fmt.Errorf("Couldn't find correct action to take. Please double check your input is correct. ")
	}

	return &conn, nil

}