package helper

import (
	"errors"
	"net/http"
	"strconv"

	"github.com/gorilla/mux"
)

// ExtractIdFromRequest takes as input a request object
// and extracts an id from it
func ExtractIDFromRequest(r *http.Request) (uint32, error) {
	vars := mux.Vars(r)
	id, err := strconv.Atoi(vars["id"])
	if err != nil {
		return 0, err
	}
	processedID := uint32(id)
	return processedID, nil
}

// ProcessMalformedRequest handles aggregated errors occurring from interactions with various external services
func ProcessMalformedRequest(w http.ResponseWriter, err error) {
	var mr *MalformedRequest
	if errors.As(err, &mr) {
		http.Error(w, mr.Msg, mr.Status)
	} else {
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
	}
}