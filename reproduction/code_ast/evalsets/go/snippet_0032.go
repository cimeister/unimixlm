package anonymous

const ProviderKey = "anonymous"

func FromJson(data []byte) (*Provider, error) {
	return &Provider{}, nil
}