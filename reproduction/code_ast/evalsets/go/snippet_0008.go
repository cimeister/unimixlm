package easypost

import "time"

type Tracker struct {
	Id                string            `json:"id,omitempty"`                //Unique identifier, begins with "trk_"
	Object            string            `json:"object,omitempty"`            //"Tracker"
	Mode              string            `json:"mode,omitempty"`              //"test" or "production"
	Tracking_Code     string            `json:"tracking_code,omitempty"`     //The tracking code provided by the carrier
	Status            string            `json:"status,omitempty"`            //The current status of the package, possible values are "unknown", "pre_transit", "in_transit", "out_for_delivery", "delivered", "available_for_pickup", "return_to_sender", "failure", "cancelled" or "error"
	Signed_By         string            `json:"signed_by,omitempty"`         //The name of the person who signed for the package (if available)
	Weight            float32           `json:"weight,omitempty"`            //The weight of the package as measured by the carrier in ounces (if available)
	Est_Delivery_Date time.Time         `json:"est_delivery_date,omitempty"` //The estimated delivery date provided by the carrier (if available)
	Shipment_Id       string            `json:"shipment_id,omitempty"`       //The id of the EasyPost Shipment object associated with the Tracker (if any)
	Carrier           string            `json:"carrier,omitempty"`           //The name of the carrier handling the shipment
	Tracking_Details  []*TrackingDetail `json:"tracking_details,omitempty"`  //Array of the associated TrackingDetail objects
	Carrier_Detail    *CarrierDetail    `json:"carrier_detail,omitempty"`    //The associated CarrierDetail object (if available)
	Public_Url        string            `json:"public_url,omitempty"`        //URL to a publicly-accessible html page that shows tracking details for this tracker
	Fees              []*Fee            `json:"fees,omitempty"`              //Array of the associated Fee objects
	Created_At        time.Time         `json:"created_at,omitempty"`        //
	Updated_At        time.Time         `json:"updated_at,omitempty"`        //

}

type TrackingDetail struct {
	Object            string            `json:"object,omitempty"`            //"TrackingDetail"
	Message           string            `json:"message,omitempty"`           //Description of the scan event
	Status            string            `json:"status,omitempty"`            //Status of the package at the time of the scan event, possible values are "unknown", "pre_transit", "in_transit", "out_for_delivery", "delivered", "available_for_pickup", "return_to_sender", "failure", "cancelled" or "error"
	Datetime          time.Time         `json:"datetime,omitempty"`          //The timestamp when the tracking scan occurred
	Source            string            `json:"source,omitempty"`            //The original source of the information for this scan event, usually the carrier
	Tracking_Location *TrackingLocation `json:"tracking_location,omitempty"` //The location associated with the scan event
}

type TrackingLocation struct {
	Object  string `json:"object,omitempty"`  //"TrackingLocation"
	City    string `json:"city,omitempty"`    //The city where the scan event occurred (if available)
	State   string `json:"state,omitempty"`   //The state where the scan event occurred (if available)
	Country string `json:"country,omitempty"` //The country where the scan event occurred (if available)
	Zip     string `json:"zip,omitempty"`     //The postal code where the scan event occurred (if available)
}

type CarrierDetail struct {
	Object                  string    `json:"object,omitempty"`                  //"CarrierDetail"
	Service                 string    `json:"service,omitempty"`                 //The service level the associated shipment was shipped with (if available)
	Container_Type          string    `json:"container_type,omitempty"`          //The type of container the associated shipment was shipped in (if available)
	Est_Delivery_Date_Local time.Time `json:"est_delivery_date_local,omitempty"` //The estimated delivery date as provided by the carrier, in the local time zone (if available)
	Est_Delivery_Time_Local time.Time `json:"est_delivery_time_local,omitempty"` //The estimated delivery time as provided by the carrier, in the local time zone (if available)

}