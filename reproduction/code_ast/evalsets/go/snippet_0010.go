package main

type TenMEvents struct {
	CmdbCi           string `json:"u_cmdb_ci,omitempty"`
	CiName           string `json:"u_cmdb_ci_name,omitempty"`
	CorrelationID    string `json:"u_correlation_id,omitempty"`
	Description      string `json:"u_description,omitempty"`
	Incident         string `json:"u_incident,omitempty"`
	ShortDescription string `json:"u_short_description,omitempty"`
	SysId            string `json:"sys_id,omitempty"`
}