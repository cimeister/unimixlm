package server

import (
	"mime/multipart"

	"github.com/go-playground/validator/v10"
)

type MergeForm struct {
	PDF []*multipart.FileHeader `form:"file[]" binding:"required,min=2"`
}

type SplitForm struct {
	PDF   *multipart.FileHeader `form:"file" binding:"required,len=1"`
	Pages []string              `form:"pages[]" binding:"required"`
}

type CompressForm struct {
	PDF *multipart.FileHeader `form:"file" binding:"required,len=1"`
}

type RotateForm struct {
	PDF      *multipart.FileHeader `form:"file" binding:"required"`
	Rotation int                   `form:"rotation" binding:"required,divisibleby90"`
	Pages    []string              `form:"pages[]" binding:"required"`
}

var divisibleBy90 validator.Func = func(fl validator.FieldLevel) bool {
	return fl.Field().Int()%90 == 0
}