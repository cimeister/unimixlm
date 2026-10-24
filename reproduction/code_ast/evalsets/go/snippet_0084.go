package util

import (
	"bytes"
	"encoding/xml"
	"errors"
	"fmt"
	"regexp"
	"strconv"
)

/*
xml to map:XmlToMapString
*/

type Node struct {
	dup   bool   // is member of a list
	attr  bool   // is an attribute
	key   string // XML tag
	val   string // element value
	nodes []*Node
}

func XmlToXmlMap(doc string, recast ...bool) (XmlMap, error) {
	var r bool
	if len(recast) == 1 {
		r = recast[0]
	}
	n, err := DocToTree(doc)
	if err != nil {
		return nil, err
	}

	m := make(map[string]interface{})
	m[n.key] = n.treeToMap(r)
	xmlMap := make(XmlMap)

	if x, ok := m["xml"].(map[string]interface{}); ok {
		for k, v := range x {
			xmlMap[k] = ToString(v)
		}
	}

	return xmlMap, nil
}

func XmlToMapString(doc string, recast ...bool) (map[string]string, error) {
	var r bool
	if len(recast) == 1 {
		r = recast[0]
	}
	n, err := DocToTree(doc)
	if err != nil {
		return nil, err
	}

	m := make(map[string]string)
	m[n.key] = n.treeToMap(r).(string)

	return m, nil
}

// XmlToMap - convert an XML doc into a map[string]interface{}.
// (This is analogous to unmarshalling a JSON string to map[string]interface{} using json.Unmarshal().)
//	If the optional argument 'recast' is 'true', then values will be converted to boolean or float64 if possible.
//	Note: recasting is only applied to element values, not attribute values.
func XmlToMap(doc string, recast ...bool) (map[string]interface{}, error) {
	var r bool
	if len(recast) == 1 {
		r = recast[0]
	}
	n, err := DocToTree(doc)
	if err != nil {
		return nil, err
	}

	m := make(map[string]interface{})
	m[n.key] = n.treeToMap(r)

	return m, nil
}

// DocToTree - convert an XML doc into a tree of nodes.
func DocToTree(doc string) (*Node, error) {
	// xml.Decoder doesn't properly handle whitespace in some doc
	// see songTextString.xml test case ...
	reg, _ := regexp.Compile("[ \t\n\r]*<")
	doc = reg.ReplaceAllString(doc, "<")

	b := bytes.NewBufferString(doc)
	p := xml.NewDecoder(b)
	n, berr := xmlToTree("", nil, p)
	if berr != nil {
		return nil, berr
	}

	return n, nil
}

// xmlToTree - load a 'clean' XML doc into a tree of *Node.
func xmlToTree(skey string, a []xml.Attr, p *xml.Decoder) (*Node, error) {
	n := new(Node)
	n.nodes = make([]*Node, 0)

	if skey != "" {
		n.key = skey
		if len(a) > 0 {
			for _, v := range a {
				na := new(Node)
				na.attr = true
				na.key = `-` + v.Name.Local
				na.val = v.Value
				n.nodes = append(n.nodes, na)
			}
		}
	}
	for {
		t, err := p.Token()
		if err != nil {
			return nil, err
		}
		switch t.(type) {
		case xml.StartElement:
			tt := t.(xml.StartElement)
			// handle root
			if n.key == "" {
				n.key = tt.Name.Local
				if len(tt.Attr) > 0 {
					for _, v := range tt.Attr {
						na := new(Node)
						na.attr = true
						na.key = `-` + v.Name.Local
						na.val = v.Value
						n.nodes = append(n.nodes, na)
					}
				}
			} else {
				nn, nnerr := xmlToTree(tt.Name.Local, tt.Attr, p)
				if nnerr != nil {
					return nil, nnerr
				}
				n.nodes = append(n.nodes, nn)
			}
		case xml.EndElement:
			// scan n.nodes for duplicate n.key values
			n.markDuplicateKeys()
			return n, nil
		case xml.CharData:
			tt := string(t.(xml.CharData))
			if len(n.nodes) > 0 {
				nn := new(Node)
				nn.key = "#text"
				nn.val = tt
				n.nodes = append(n.nodes, nn)
			} else {
				n.val = tt
			}
		default:
			// noop
		}
	}
	// Logically we can't get here, but provide an error message anyway.
	return nil, errors.New("Unknown parse error in xmlToTree() for: " + n.key)
}

// (*Node)markDuplicateKeys - set node.dup flag for loading map[string]interface{}.
func (n *Node) markDuplicateKeys() {
	l := len(n.nodes)
	for i := 0; i < l; i++ {
		if n.nodes[i].dup {
			continue
		}
		for j := i + 1; j < l; j++ {
			if n.nodes[i].key == n.nodes[j].key {
				n.nodes[i].dup = true
				n.nodes[j].dup = true
			}
		}
	}
}

// (*Node)treeToMap - convert a tree of nodes into a map[string]interface{}.
//	(Parses to map that is structurally the same as from json.Unmarshal().)
// Note: root is not instantiated; call with: "m[n.key] = n.treeToMap(recast)".
func (n *Node) treeToMap(r bool) interface{} {
	if len(n.nodes) == 0 {
		return recast(n.val, r)
	}

	m := make(map[string]interface{}, 0)
	for _, v := range n.nodes {
		// just a value
		if !v.dup && len(v.nodes) == 0 {
			m[v.key] = recast(v.val, r)
			continue
		}

		// a list of values
		if v.dup {
			var a []interface{}
			if vv, ok := m[v.key]; ok {
				a = vv.([]interface{})
			} else {
				a = make([]interface{}, 0)
			}
			a = append(a, v.treeToMap(r))
			m[v.key] = interface{}(a)
			continue
		}

		// it's a unique key
		m[v.key] = v.treeToMap(r)
	}

	return interface{}(m)
}

// recast - try to cast string values to bool or float64
func recast(s string, r bool) interface{} {
	if r {
		// handle numeric strings ahead of boolean
		if f, err := strconv.ParseFloat(s, 64); err == nil {
			return interface{}(f)
		}
		// ParseBool treats "1"==true & "0"==false
		if b, err := strconv.ParseBool(s); err == nil {
			return interface{}(b)
		}
	}
	return interface{}(s)
}

// WriteMap - dumps the map[string]interface{} for examination.
//	'offset' is initial indentation count; typically: WriteMap(m).
//	NOTE: with XML all element types are 'string'.
//	But code written as generic for use with maps[string]interface{} values from json.Unmarshal().
//	Or it can handle a XmlToMap(doc,true) result where values have be recast'd.
func WriteMap(m interface{}, offset ...int) string {
	var indent int
	if len(offset) == 1 {
		indent = offset[0]
	}

	var s string
	switch m.(type) {
	case nil:
		return "[nil] nil"
	case string:
		return "[string] " + m.(string)
	case float64:
		return "[float64] " + strconv.FormatFloat(m.(float64), 'e', 2, 64)
	case bool:
		return "[bool] " + strconv.FormatBool(m.(bool))
	case []interface{}:
		s += "[[]interface{}]"
		for i, v := range m.([]interface{}) {
			s += "\n"
			for i := 0; i < indent; i++ {
				s += "  "
			}
			s += "[item: " + strconv.FormatInt(int64(i), 10) + "]"
			switch v.(type) {
			case string, float64, bool:
				s += "\n"
			default:
				// noop
			}
			for i := 0; i < indent; i++ {
				s += "  "
			}
			s += WriteMap(v, indent+1)
		}
	case map[string]interface{}:
		for k, v := range m.(map[string]interface{}) {
			s += "\n"
			for i := 0; i < indent; i++ {
				s += "  "
			}
			// s += "[map[string]interface{}] "+k+" :"+WriteMap(v,indent+1)
			s += k + " :" + WriteMap(v, indent+1)
		}
	default:
		// shouldn't ever be here ...
		s += fmt.Sprintf("unknown type for: %v", m)
	}
	return s
}