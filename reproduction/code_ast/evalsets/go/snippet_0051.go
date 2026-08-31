package ottaconf

import "encoding/xml"

// Event describes the individual events occuring at the conference
type Event struct {
	id          int
	room        *Room
	day         *Date
	start       Time
	duration    Time
	title       string
	subtitle    string
	slug        string
	track       string
	eventType   string
	language    string
	abstract    string
	description string
	people      []*Person
	links       []*Link
}

// UnmarshalXML provides an interface to unmarshal XML encoded data about an event into the internal representation.
func (e *Event) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	var data struct {
		XMLName     xml.Name  `xml:"event"`
		ID          int       `xml:"id,attr"`
		Room        string    `xml:"room"`
		Start       string    `xml:"start"`
		Duration    string    `xml:"duration"`
		Slug        string    `xml:"slug"`
		Title       string    `xml:"title"`
		Subtitle    string    `xml:"subtitle"`
		Track       string    `xml:"track"`
		Type        string    `xml:"type"`
		Language    string    `xml:"language"`
		Abstract    string    `xml:"abstract"`
		Description string    `xml:"description"`
		People      []*Person `xml:"persons>person"`
		Links       []*Link   `xml:"links>link"`
	}
	err := d.DecodeElement(&data, &start)
	if err != nil {
		return err
	}
	e.id = data.ID
	e.room = &Room{name: data.Room}
	e.start, err = AtoTime(data.Start)
	if err != nil {
		return err
	}
	e.duration, err = AtoTime(data.Duration)
	if err != nil {
		return err
	}
	e.slug = data.Slug
	e.title = data.Title
	e.subtitle = data.Subtitle
	e.track = data.Track
	e.eventType = data.Type
	e.language = data.Language
	e.abstract = data.Abstract
	e.description = data.Description
	e.people = data.People
	e.links = data.Links
	return nil
}

// String returns a printable description of the event, in this case, the title
func (e *Event) String() string {
	return e.title
}

// Title returns the title of the event
func (e *Event) Title() string {
	return e.title
}

// ID returns the unique id used to reference the event within the XML schedule.
func (e *Event) ID() int {
	return e.id
}

// Room returns information on the room the event is hosted in.
func (e *Event) Room() *Room {
	return e.room
}

// Date returns the date on which the event takes place
func (e *Event) Date() *Date {
	return e.day
}

// StartTime returns the time at which the event is scheduled to start
func (e *Event) StartTime() Time {
	return e.start
}

// Duration returns the length of time the event is scheduled to run for
func (e *Event) Duration() Time {
	return e.duration
}

// Subtitle returns a short subtitle for the event
func (e *Event) Subtitle() string {
	return e.subtitle
}

// Slug returns a unique string by which to identify the event
func (e *Event) Slug() string {
	return e.slug
}

// Track returns the track of which this event is part of
func (e *Event) Track() string {
	return e.track
}

// Type returns what sort of event is being represented
func (e *Event) Type() string {
	return e.eventType
}

// Abstract returns a short paragraph about the event
func (e *Event) Abstract() string {
	return e.abstract
}

// Description returns a detailed description of the event
func (e *Event) Description() string {
	return e.description
}

// People returns a list of people hosting/associated with the event
func (e *Event) People() []*Person {
	people := make([]*Person, len(e.people))
	copy(people, e.people)
	return people
}

// Links returns a list of links to material associated with the event
func (e *Event) Links() []*Link {
	links := make([]*Link, len(e.links))
	copy(links, e.links)
	return links
}