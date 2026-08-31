package k8s

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	Water                    ResourceType = "waters"
	Deployment               ResourceType = "deployments"
	Stone                    ResourceType = "stones"
	StatefulSet              ResourceType = "statefulsets"
	StatefulSet1             ResourceType = "statefulset1s"
	DaemonSet                ResourceType = "daemonsets"
	Injector                 ResourceType = "injectors"
	Pod                      ResourceType = "pods"
	Job                      ResourceType = "jobs"
	CronJob                  ResourceType = "cronjobs"
	ReplicaSet               ResourceType = "replicasets"
	Event                    ResourceType = "events"
	Node                     ResourceType = "nodes"
	ConfigMap                ResourceType = "configmaps"
	Secret                   ResourceType = "secrets"
	ResourceQuota            ResourceType = "resourcequotas"
	Service                  ResourceType = "services"
	Ingress                  ResourceType = "ingresses"
	NetworkPolicy            ResourceType = "networkpolicies"
	HorizontalPodAutoscaler  ResourceType = "horizontalpodautoscalers"
	CustomResourceDefinition ResourceType = "customresourcedefinitions"
	PersistentVolume         ResourceType = "persistentvolumes"
	PersistentVolumeClaims   ResourceType = "persistentvolumeclaims"
	StorageClass             ResourceType = "storageclasses"
	ServiceAccount           ResourceType = "serviceaccounts"
	Role                     ResourceType = "roles"
	ClusterRole              ResourceType = "clusterroles"
	RoleBinding              ResourceType = "rolebindings"
	Namespace                ResourceType = "namespaces"
	PodSecurityPolicie       ResourceType = "podsecuritypolicies"
	ClusterRoleBinding       ResourceType = "clusterrolebindings"
	Endpoint                 ResourceType = "endpoints"

	Metrics ResourceType = "metrics"

	// Workloads deployment resource workloads template for CaaS
	Workloads ResourceType = "workloads"

	// form render
	FormRender ResourceType = "formrenders"
	Page       ResourceType = "pages"
	Form       ResourceType = "forms"
	Field      ResourceType = "fields"

	// tenant for PaaS
	BaseDepartment ResourceType = "basedepartments"
	BaseTenant     ResourceType = "basetenants"
	BaseRole       ResourceType = "baseroles"
	BaseUser       ResourceType = "baseusers"
	BaseRoleUser   ResourceType = "baseroleusers"

	// network for container ovn/ovs control plant
	IP                          ResourceType = "ips"
	SubNet                      ResourceType = "subnets"
	Vlan                        ResourceType = "vlans"
	NetworkAttachmentDefinition ResourceType = "network-attachment-definitions"

	//tekton
	Pipeline         ResourceType = "pipelines"
	PipelineRun      ResourceType = "pipelineruns"
	Task             ResourceType = "tasks"
	TaskRun          ResourceType = "taskruns"
	PipelineResource ResourceType = "pipelineresources"
	TektonGraph      ResourceType = "tektongraphs"
	TektonConfig     ResourceType = "tektonconfigs"
	TektonWebHook    ResourceType = "tektonwebhooks"
	TektonStore      ResourceType = "tektonstores"

	//Istio  NetWorking
	Gateway         ResourceType = "gateways"
	DestinationRule ResourceType = "destinationrules"
	ServiceEntry    ResourceType = "serviceentries"
	Sidecar         ResourceType = "sidecars"
	VirtualService  ResourceType = "virtualservices"
	WorkloadEntry   ResourceType = "workloadentries"
)

type Op = string

var (
	tektonResource = map[string]map[string]bool{
		Pipeline: {
			"create": false,
			"update": false,
			"delete": false,
		},
	}
)

type IGVRMaps interface {
	Subscribe(include ...string) []Resource
}

type groupVersionCollection map[ResourceType]schema.GroupVersionResource

func (c *groupVersionCollection) Subscribe(include ...string) []Resource {
	result := make([]Resource, 0)
	if len(include) == 0 {
		return result
	}
	for key, value := range *c {
		for _, item := range include {
			if item == key {
				result = append(result, Resource{key, value})
			}
		}
	}
	return result
}

// describe resource collection
var GVRMaps IGVRMaps = &groupVersionCollection{
	Water:        {Group: "nuwa.nip.io", Version: "v1", Resource: "waters"},
	Stone:        {Group: "nuwa.nip.io", Version: "v1", Resource: "stones"},
	StatefulSet1: {Group: "nuwa.nip.io", Version: "v1", Resource: "statefulsets"},
	Injector:     {Group: "nuwa.nip.io", Version: "v1", Resource: "injectors"},

	// since 0.2.0 the group version change to yamecloud.io
	Workloads: {Group: "yamecloud.io", Version: "v1", Resource: "workloads"},

	// PaaS RBAC
	BaseTenant:     {Group: "yamecloud.io", Version: "v1", Resource: "basetenants"},
	BaseDepartment: {Group: "yamecloud.io", Version: "v1", Resource: "basedepartments"},
	BaseRole:       {Group: "yamecloud.io", Version: "v1", Resource: "baseroles"},
	BaseUser:       {Group: "yamecloud.io", Version: "v1", Resource: "baseusers"},
	BaseRoleUser:   {Group: "yamecloud.io", Version: "v1", Resource: "baseroleusers"},

	StatefulSet: {Group: "apps", Version: "v1", Resource: "statefulsets"},
	DaemonSet:   {Group: "apps", Version: "v1", Resource: "daemonsets"},
	ReplicaSet:  {Group: "apps", Version: "v1", Resource: "replicasets"},
	Deployment:  {Group: "apps", Version: "v1", Resource: "deployments"},

	Job:     {Group: "batch", Version: "v1", Resource: "jobs"},
	CronJob: {Group: "batch", Version: "v1beta1", Resource: "cronjobs"},

	Pod:                    {Group: "", Version: "v1", Resource: "pods"},
	Node:                   {Group: "", Version: "v1", Resource: "nodes"},
	Event:                  {Group: "", Version: "v1", Resource: "events"},
	ConfigMap:              {Group: "", Version: "v1", Resource: "configmaps"},
	Secret:                 {Group: "", Version: "v1", Resource: "secrets"},
	ResourceQuota:          {Group: "", Version: "v1", Resource: "resourcequotas"},
	Service:                {Group: "", Version: "v1", Resource: "services"},
	Namespace:              {Group: "", Version: "v1", Resource: "namespaces"},
	PersistentVolume:       {Group: "", Version: "v1", Resource: "persistentvolumes"},
	PersistentVolumeClaims: {Group: "", Version: "v1", Resource: "persistentvolumeclaims"},
	ServiceAccount:         {Group: "", Version: "v1", Resource: "serviceaccounts"},
	Endpoint:               {Group: "", Version: "v1", Resource: "endpoints"},

	Ingress:                  {Group: "networking.k8s.io", Version: "v1", Resource: "ingresses"},
	NetworkPolicy:            {Group: "networking.k8s.io", Version: "v1", Resource: "networkpolicies"},
	HorizontalPodAutoscaler:  {Group: "autoscaling", Version: "v2beta1", Resource: "horizontalpodautoscalers"},
	CustomResourceDefinition: {Group: "apiextensions.k8s.io", Version: "v1", Resource: "customresourcedefinitions"},

	StorageClass: {Group: "storage.k8s.io", Version: "v1", Resource: "storageclasses"},

	ClusterRole:        {Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterroles"},
	Role:               {Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "roles"},
	ClusterRoleBinding: {Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterrolebindings"},
	RoleBinding:        {Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "rolebindings"},

	IP:     {Group: "kubeovn.io", Version: "v1", Resource: "ips"},
	SubNet: {Group: "kubeovn.io", Version: "v1", Resource: "subnets"},
	Vlan:   {Group: "kubeovn.io", Version: "v1", Resource: "vlans"},

	NetworkAttachmentDefinition: {Group: "k8s.cni.cncf.io", Version: "v1", Resource: "network-attachment-definitions"},

	// tekton.dev resource view
	Pipeline:         {Group: "tekton.dev", Version: "v1alpha1", Resource: "pipelines"},
	PipelineRun:      {Group: "tekton.dev", Version: "v1alpha1", Resource: "pipelineruns"},
	Task:             {Group: "tekton.dev", Version: "v1alpha1", Resource: "tasks"},
	TaskRun:          {Group: "tekton.dev", Version: "v1alpha1", Resource: "taskruns"},
	PipelineResource: {Group: "tekton.dev", Version: "v1alpha1", Resource: "pipelineresources"},
	TektonGraph:      {Group: "yamecloud.io", Version: "v1", Resource: "tektongraphs"},
	TektonConfig:     {Group: "", Version: "v1", Resource: "secrets"},
	TektonWebHook:    {Group: "yamecloud.io", Version: "v1", Resource: "tektonwebhooks"},
	TektonStore:      {Group: "yamecloud.io", Version: "v1", Resource: "tektonstores"},

	PodSecurityPolicie: {Group: "policy", Version: "v1beta1", Resource: "podsecuritypolicies"},

	//Istio Networking
	Gateway:         {Group: "networking.istio.io", Version: "v1beta1", Resource: "gateways"},
	DestinationRule: {Group: "networking.istio.io", Version: "v1beta1", Resource: "destinationrules"},
	ServiceEntry:    {Group: "networking.istio.io", Version: "v1beta1", Resource: "serviceentries"},
	Sidecar:         {Group: "networking.istio.io", Version: "v1beta1", Resource: "sidecars"},
	VirtualService:  {Group: "networking.istio.io", Version: "v1beta1", Resource: "virtualservices"},
	WorkloadEntry:   {Group: "networking.istio.io", Version: "v1beta1", Resource: "workloadentries"},
}