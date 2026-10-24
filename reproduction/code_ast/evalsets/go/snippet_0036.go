package sync

// Sync : Sync interface for the service
type Sync interface {
	Start() error
	Stop() error
}