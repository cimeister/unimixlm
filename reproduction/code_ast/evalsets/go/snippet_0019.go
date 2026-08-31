package db

import (
	"errors"
	"log"
	"time"

	"gopkg.in/mgo.v2"
	"gopkg.in/mgo.v2/bson"

	"github.com/changer/khabar/utils"
)

var Conn *MConn

func Convert(doc utils.M, out interface{}) {
	stream, err := bson.Marshal(doc)
	if err == nil {
		bson.Unmarshal(stream, out)
	} else {
		panic(err)
	}
}

type MConn struct {
	Session *mgo.Session
	Dbname  string
}

func (self *MConn) getCursor(session *mgo.Session, table string, query utils.M) *mgo.Query {

	fields, err1 := query["fields"].(utils.M)
	delete(query, "fields")
	if !err1 {
		fields = utils.M{}
	}

	sort, err2 := query["sort"].(string)
	delete(query, "sort")
	if !err2 {
		sort = "$natural"
	}

	skip, err3 := query["skip"].(int)
	delete(query, "skip")
	if !err3 {
		skip = 0
	}

	limit, err4 := query["limit"].(int)
	delete(query, "limit")
	if !err4 {
		limit = 0
	}

	cursor := self.GetCursor(session, table, query)
	return cursor.Limit(limit).Skip(skip).Sort(sort).Select(fields)
}

type MapReduce mgo.MapReduce

func (self *MConn) MapReduce(session *mgo.Session, table string, query utils.M, result interface{}, job *MapReduce) (*mgo.MapReduceInfo, error) {
	db := session.DB(self.Dbname)

	coll := db.C(table)
	realJob := mgo.MapReduce{Map: job.Map, Reduce: job.Reduce, Finalize: job.Finalize, Scope: job.Scope, Verbose: true}
	return coll.Find(query).MapReduce(&realJob, result)
}

func (self *MConn) DropIndex(table string, key ...string) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	coll := db.C(table)
	return coll.DropIndex(key...)
}

func (self *MConn) DropIndices(table string) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	collection := db.C(table)
	indexes, err := collection.Indexes()
	if err == nil {
		for _, index := range indexes {
			err = collection.DropIndex(index.Key...)
			if err != nil {
				return err
			}
		}
	}

	if err != nil {
		panic(err)
	}
	return nil
}

func (self *MConn) findAndApply(table string, query utils.M, change mgo.Change, result interface{}) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	change.ReturnNew = true

	coll := db.C(table)
	_, err := coll.Find(query).Apply(change, result)
	if err != nil {
		log.Println("Error Applying Changes", table, err)
	}
	return err
}

func (self *MConn) FindAndUpsert(table string, query utils.M, doc utils.M, result interface{}) error {
	change := mgo.Change{
		Update: doc,
		Upsert: true,
	}
	return self.findAndApply(table, query, change, result)
}

func (self *MConn) FindAndUpdate(table string, query utils.M, doc utils.M, result interface{}) error {
	change := mgo.Change{
		Update: doc,
		Upsert: false,
	}
	return self.findAndApply(table, query, change, result)
}

func (self *MConn) EnsureIndex(table string, index mgo.Index) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	coll := db.C(table)
	return coll.EnsureIndex(index)
}

func (self *MConn) GetCursor(session *mgo.Session, table string, query utils.M) *mgo.Query {
	db := session.DB(self.Dbname)

	coll := db.C(table)
	out := coll.Find(query)

	return out
}

func (self *MConn) Get(session *mgo.Session, table string, query utils.M) *mgo.Iter {
	return self.getCursor(session, table, query).Iter()
}

func (self *MConn) HintedGetOne(table string, query utils.M, result interface{}, hint string) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	defer session.Close()

	cursor := self.getCursor(session, table, query).Hint(hint)
	err := cursor.One(result)
	if err != nil {
		log.Println("Error fetching", table, err)
	}

	return err
}

func (self *MConn) GetOne(table string, query utils.M, result interface{}) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	defer session.Close()

	cursor := self.getCursor(session, table, query)
	err := cursor.One(result)
	if err != nil {
		//log.Println("Error fetching", table, err)
	}

	return err
}

func (self *MConn) HintedCount(table string, query utils.M, hint string) int {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	defer session.Close()

	cursor := self.getCursor(session, table, query).Select(utils.M{"_id": 1}).Hint(hint)
	count, err := cursor.Count()
	if err != nil {
		log.Println("Error Counting", table, err)
	}

	return count
}

func (self *MConn) Count(table string, query utils.M) int {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	defer session.Close()

	cursor := self.getCursor(session, table, query).Select(utils.M{"_id": 1})
	count, err := cursor.Count()
	if err != nil {
		log.Println("Error Counting", table, err)
	}

	return count
}

func (self *MConn) Upsert(table string, query utils.M, doc utils.M) error {

	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	var err error
	if len(doc) == 0 {
		err = errors.New(
			"Empty upsert is blocked. Refer to " +
				"https://github.com/Simversity/blackjack/issues/1051",
		)
	} else {
		coll := db.C(table)
		_, err = coll.Upsert(query, doc)
	}

	if err != nil {
		log.Println("Error Upserting:", table, err)
	}
	return err
}

func AlterDoc(doc *utils.M, operator string, operation utils.M) {
	spec := *doc
	if spec[operator] != nil {
		op, _ := spec[operator].(utils.M)
		for key, value := range op {
			operation[key] = value
		}
	}
	spec[operator] = operation
}

func (self *MConn) Update(table string, query utils.M, doc utils.M) error {

	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	coll := db.C(table)
	var update_err error
	if len(doc) == 0 {
		update_err = errors.New(
			"Empty Update is blocked. Refer to " +
				"https://github.com/Simversity/blackjack/issues/1051",
		)
	} else {
		AlterDoc(&doc, "$set", utils.M{"updated_on": utils.EpochNow()})
		_, update_err = coll.UpdateAll(query, doc)
	}

	if update_err != nil {
		log.Println("Error Updating:", table, update_err)
	}
	return update_err
}

func (self *MConn) Delete(table string, query utils.M) error {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	var delete_err error

	coll := db.C(table)

	_, delete_err = coll.RemoveAll(query)

	if delete_err != nil {
		log.Println("Error Deleting:", table, delete_err)
	}

	return delete_err
}

func InArray(key string, arrays ...[]string) bool {
	for _, val := range arrays {
		for _, one := range val {
			if key == one {
				return true
			}
		}
	}
	return false
}

func (self *MConn) Insert(table string, arguments ...interface{}) (_id string) {
	//Create a Session Copy and be responsible for Closing it.
	session := self.Session.Copy()
	db := session.DB(self.Dbname)
	defer session.Close()

	var out interface{}
	if len(arguments) > 1 {
		out = arguments[1]
	} else {
		out = nil
	}

	doc := arguments[0]

	coll := db.C(table)
	err := coll.Insert(doc)
	if err != nil {
		panic(err)
	}

	if out != nil {
		stream, merr := bson.Marshal(doc)
		if merr == nil {
			bson.Unmarshal(stream, out)
		}
	}

	return
}

func (self *MConn) Aggregate(session *mgo.Session, table string, doc []utils.M) *mgo.Pipe {
	//Create a Session Copy and be responsible for Closing it.
	db := session.DB(self.Dbname)

	coll := db.C(table)
	return coll.Pipe(doc)
}

var cachedConnections = map[string]*mgo.Session{}

func GetConn(db_name string, address string, creds ...string) *MConn {
	//Check if the connection has been stored already.
	_, ok := cachedConnections[db_name]
	if !ok {
		var username, password string

		if len(creds) > 0 {
			username = creds[0]
			if len(creds) > 1 {
				password = creds[1]
			}
		}

		// Timeout is the amount of time to wait for a server to respond when
		// first connecting and on follow up operations in the session. If
		// timeout is zero, the call may block forever waiting for a connection
		// to be established.

		// FailFast will cause connection and query attempts to fail faster when
		// the server is unavailable, instead of retrying until the configured
		// timeout period. Note that an unavailable server may silently drop
		// packets instead of rejecting them, in which case it's impossible to
		// distinguish it from a slow server, so the timeout stays relevant.

		info := mgo.DialInfo{
			Addrs:    []string{address},
			Timeout:  60 * time.Second,
			FailFast: true,
			Database: db_name,
			Username: username,
			Password: password,
		}

		session, err := mgo.DialWithInfo(&info)
		if err != nil {
			panic(err)
		}

		cachedConnections[db_name] = session
	}

	//Save the Session for Later use.
	session := cachedConnections[db_name]

	//Return only a Session & the name. Let the Consumer make a Session.Copy()
	//to ensure that database state is resumed.

	return &MConn{session, db_name}
}