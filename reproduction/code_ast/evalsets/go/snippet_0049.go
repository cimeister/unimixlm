// Copyright © 2020 lvzhihao <<EMAIL>>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package cmd

import (
	"bytes"
	"net/http"
	"os"
	"os/exec"
	"strings"

	"github.com/labstack/echo"
	"github.com/lvzhihao/goutils"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

type ApiResult struct {
	Code string      `json:"code"` //code: 000000
	Data interface{} `json:"data"` //result data
}

// webhookCmd represents the webhook command
var webhookCmd = &cobra.Command{
	Use:   "webhook",
	Short: "webhook api",
	Long:  `webhook api`,
	Run: func(cmd *cobra.Command, args []string) {
		var trigger chan string
		var logger *zap.Logger
		if os.Getenv("DEBUG") == "true" {
			logger, _ = zap.NewDevelopment()
		} else {
			logger, _ = zap.NewProduction()
		}
		defer logger.Sync()
		// app.Logger.SetLevel(log.INFO)
		app := goutils.NewEcho()
		trigger = make(chan string, 5)

		app.POST("/webhook", func(ctx echo.Context) error {
			if strings.Compare(viper.GetString("TOKEN"), ctx.QueryParam("token")) != 0 {
				return ctx.NoContent(http.StatusNotFound)
			}
			trigger <- "updates" //todo body info
			logger.Info("New Trigger", zap.Int("queueCount", len(trigger)))
			return ctx.JSON(http.StatusOK, ApiResult{
				Code: "000000",
				Data: "success",
			})
		})

		go func() {
			for {
				select {
				case data := <-trigger:
					logger.Info("Run Trigger", zap.String("data", data), zap.Int("queueCount", len(trigger)))
					cmd := exec.Command("/bin/sh", "deploy.sh")
					cmd.Dir = viper.GetString("CMD_PATH")
					var out bytes.Buffer
					cmd.Stdout = &out
					if err := cmd.Run(); err != nil {
						logger.Error("Cmd error", zap.Error(err), zap.String("output", out.String()))
					} else {
						logger.Info("Cmd Success")
						logger.Debug("Cmd Success Info", zap.String("output", out.String()))
					}
				}
			}
		}()

		// graceful shutdown
		goutils.EchoStartWithGracefulShutdown(app, ":3000")
	},
}

func init() {
	rootCmd.AddCommand(webhookCmd)

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// webhookCmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// webhookCmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}