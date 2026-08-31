package router

import (
	"ginDemo/backend/server"
	"github.com/gin-gonic/gin"
)

func Login(r *gin.Engine) {

	r.POST("/login", server.LoginServer)

}

func GetMessage(r *gin.Engine) {

	r.GET("/toget", server.GetMsg)

}