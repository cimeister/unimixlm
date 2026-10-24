package webserver;

/**
 * Created by matthias on 21.11.17.
 */
public class RestResponse {
    private String message;

    public RestResponse() {
    }

    public RestResponse(String message) {
        this.message = message;
    }

    public String getMessage() {
        return message;
    }
}