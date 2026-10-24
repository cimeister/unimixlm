package models

import (
	"fmt"

	"github.com/astaxie/beego/orm"
)

type User struct {
	Id   int
	Name string `orm:"size(100)"`
}

type TestModel struct {
}

func (m *TestModel) SaveTest() string {
	o := orm.NewOrm()
	user := User{Name: "test"}
	id, err := o.Insert(&user)
	return fmt.Sprintf("Id: %d, ERR: %v\n", id, err)
}

func (m *TestModel) UpdateTest() string {
	o := orm.NewOrm()
	user := User{Name: "test2"}
	num, err := o.Update(&user)
	return fmt.Sprintf("NUM: %d. ERR: %v\n", num, err)
}

func (m *TestModel) ReadTest() string {
	o := orm.NewOrm()
	user := User{Id: 1}
	err := o.Read(&user)
	return fmt.Sprintf("ERR: %v\n", err)
}

func (m *TestModel) DeleteTest() string {
	o := orm.NewOrm()
	user := User{Id: 1, Name: "test2"}
	num, err := o.Delete(&user)
	return fmt.Sprintf("NUM: %d. ERR: %v\n", num, err)
}