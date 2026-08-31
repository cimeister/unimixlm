package draw

import (
	"os"
	"path/filepath"
)

var SystemInstace *System

const (
	LANGUAGE_CHINESE_ENGLISH = 0x0409
	LANGUAGE_CHINESE_SIMPLE  = 0x0804
	LANGUAGE_CHINESE_TW      = 0x0404
	LANGUAGE_CHINESE_HK      = 0x0C04
	LANGUAGE_CHINESE_SGP     = 0x1004
)

type System struct {
	languaeId int
	rootPath  string
}

func (this *System) GetLanguage() string {
	switch this.languaeId {
	case LANGUAGE_CHINESE_ENGLISH:
		return "en-us"
	case LANGUAGE_CHINESE_SIMPLE:
		return "zh-cn"
	case LANGUAGE_CHINESE_TW:
		return "zh-tw"
	case LANGUAGE_CHINESE_HK:
		return "zh-tw"
	case LANGUAGE_CHINESE_SGP:
		return "zh-tw"
	}

	return "en-us"
}

func (this *System) GetCurrentLanguage() int {
	return this.languaeId
}

func (this *System) SetCurrentLanguage(languaeId int) {
	this.languaeId = languaeId
}

func (this *System) SetRootPath(rootPath string) {
	this.rootPath = rootPath
}

func (this *System) GetRootPath() string {
	if len(this.rootPath) > 0 {
		return this.rootPath
	}
	return this.GetExePath()
}

func (this *System) GetExePath() string {
	path, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		panic(err)
	}
	return filepath.FromSlash(path)
}

func (this *System) GetSkinPath() string {
	return this.GetRootPath()
}

func (this *System) GetXmlPath() string {
	return filepath.FromSlash(this.GetRootPath() + "/xml")
}