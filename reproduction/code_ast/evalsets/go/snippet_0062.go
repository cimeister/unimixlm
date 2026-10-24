package rest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"reflect"

	"github.com/breadysimon/goless/logging"
	"github.com/breadysimon/goless/reflection"
	"github.com/breadysimon/goless/text"
	"gopkg.in/resty.v1"
)

var log *logging.Logger = logging.GetLogger()

type Client struct {
	url         string
	api         *resty.Client
	token       string
	err         error
	result      interface{}
	errorResult interface{}
	username    string
}

func NewClient(baseUrl, rootPath, username, password string) *Client {
	c := &Client{url: baseUrl + rootPath}
	c.api = resty.New()
	return c.Login(username, password)
}
func (c *Client) Error() error {
	return c.err
}

type token struct {
	Code   int    `json:"code"`
	Expire string `json:"expire"`
	Token  string `json:"token"`
}

func (c *Client) Login(username, password string) *Client {
	if c.err == nil {
		var u *url.URL
		if u, c.err = url.Parse(c.url); c.err == nil {
			url := fmt.Sprintf("%s://%s:%s/login", u.Scheme, u.Hostname(), u.Port())

			var t token
			var resp *resty.Response
			if resp, c.err = c.api.R().
				SetFormData(map[string]string{
					"username": username,
					"password": password,
				}).
				SetResult(&t).
				Post(url); c.err == nil {
				if resp.StatusCode() == http.StatusOK {
					c.token = t.Token
					// log.Info("login success: ", username)
					// log.Debug(c.token)
				} else {
					c.err = fmt.Errorf("login failed: %s", username)
				}
			}
		}
	}
	return c
}

type ErrorResult struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (c *Client) prepareRequest(body, result, errorResult interface{}) *resty.Request {
	r := c.api.R().
		SetHeader("Content-Type", "application/json").
		SetHeader("Authorization", "Bearer "+c.token)
	if body != nil {
		r = r.SetBody(body)
	}
	if result != nil {
		r = r.SetResult(result)
	}
	if errorResult != nil {
		r = r.SetError(errorResult)
	}
	return r
}

func (c *Client) checkError(err error, e *ErrorResult, resp *resty.Response, expected int) bool {
	switch {
	case err != nil:
		c.err = err
		c.errorResult = nil
		return false
	case resp.StatusCode() != expected:
		c.errorResult = e
		if e.Code == 0 {
			e.Code = resp.StatusCode()
			e.Message = resp.Status()
		}
		c.err = fmt.Errorf("failed. [%s]%s: %v", resp.Request.Method, resp.Request.URL, e)
		return false
	default:
		return true
	}
}

func (c *Client) httpPut(path string, body interface{}, result interface{}) *Client {
	var e ErrorResult
	resp, err := c.prepareRequest(body, result, &e).Put(c.url + path)

	if c.checkError(err, &e, resp, http.StatusOK) {
		c.result = result
	}
	return c
}
func (c *Client) httpPost(path string, body interface{}, result interface{}) *Client {
	var e ErrorResult
	resp, err := c.prepareRequest(body, result, &e).Post(c.url + path)

	if c.checkError(err, &e, resp, http.StatusOK) {
		c.result = result
	}
	return c
}

func (c *Client) httpGet(path string, result interface{}, query ...interface{}) *Client {
	var e ErrorResult

	req := c.prepareRequest(nil, result, &e)
	if len(query) > 0 {
		if parms, ok := query[0].(map[string]string); ok {
			req = req.SetQueryParams(parms)
		}
	}
	resp, err := req.Get(c.url + path)

	if c.checkError(err, &e, resp, http.StatusOK) {
		c.result = result
	}

	return c
}
func (c *Client) httpDelete(path string, body interface{}) *Client {
	var e ErrorResult
	resp, err := c.prepareRequest(nil, nil, &e).SetBody(body).Delete(c.url + path)

	c.checkError(err, &e, resp, http.StatusOK)
	return c
}

func (c *Client) Create(body interface{}) *Client {
	if c.err == nil {
		var result interface{}
		return c.httpPost(generateResourceName(body), body, result)
	}
	return c
}
func (c *Client) Update(body interface{}) *Client {
	if c.err == nil {
		var result interface{}
		url := fmt.Sprintf("%s/%d", generateResourceName(body), reflection.GetId(body).Int())
		return c.httpPut(url, body, result)
	}
	return c
}

func (c *Client) Find(result interface{}) *Client {
	if c.err == nil {
		resultType := reflect.TypeOf(result).String()
		if text.StartWith(resultType, "*[]") {
			return c.httpGet(generateResourceName(result), result)
		} else {

			var url string
			id := reflection.GetId(result).Int()
			if id != 0 {
				url = fmt.Sprintf("%s/%d", generateResourceName(result), id)
				return c.httpGet(url, result)
			} else {
				lst := c.findItems(result)
				if c.err == nil {
					if reflection.GetSliceLen(lst) > 0 {
						reflection.SetValue(result, reflection.GetSliceItem(lst, 0))
					}
					c.result = result

				}
			}

		}
	}
	return c
}

func composeFilter(m map[string]interface{}) string {
	s, _ := json.Marshal(m)
	return string(s)
}
func (c *Client) findItems(obj interface{}) (lst interface{}) {
	url := fmt.Sprintf("%s", generateResourceName(obj))
	parms := make(map[string]string)
	parms["filter"] = composeFilter(reflection.GetFields(obj))
	lst = reflection.MakeSlice(obj)
	c.httpGet(url, lst, parms)
	return lst
}
func (c *Client) Delete(obj interface{}) *Client {
	if c.err == nil {
		// id := reflection.GetId(obj).Int()
		// if id == 0 {
		// 	lst := c.findItems(obj)
		// 	if reflection.GetSliceLen(lst) > 0 {
		// 		id = reflection.GetId(reflection.GetSliceItem(lst, 0)).Int()
		// 	} else {
		// 		log.Info("Not found. Nothing to do for deletion.")
		// 		return c
		// 	}
		// }
		// url := fmt.Sprintf("%s/%d", generateResourceName(obj), id)

		path := fmt.Sprintf("%s/%d", generateResourceName(obj), reflection.GetId(obj).Int())
		return c.httpDelete(path, obj)

	}
	return c
}
func (c *Client) peekResult() *Client {
	if c.err == nil {
		fmt.Println(c.result)
	}
	return c
}

func (c *Client) Do(f func(c *Client)) *Client {
	if c.err == nil {
		f(c)
	}
	return c
}

func (c *Client) UserName() string {
	return "admin"
}