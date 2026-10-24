package no.nav.pus.decorator.redirect;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import no.nav.pus.decorator.config.RedirectConfig;

import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.ws.rs.core.UriBuilder;
import java.io.IOException;
import java.net.URI;

import static java.util.Optional.ofNullable;

@Slf4j
public class RedirectServlet extends HttpServlet {

    private final URI target;

    @SneakyThrows
    public RedirectServlet(RedirectConfig redirectConfig) {
        this.target = redirectConfig.to.toURI();
    }

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws IOException {
        String target = UriBuilder.fromUri(this.target).path(req.getPathInfo()).toString();
        String query = ofNullable(req.getQueryString()).map(queryString -> "?" + queryString).orElse("");
        String location = target + query;
        log.info("redirecting {} to {}", req, location);
        String s = resp.encodeRedirectURL(location);
        resp.sendRedirect(location);
    }

}