/* world.java  -  Demonstrate SSO with frontend mod_auth_saml
 * Adapted from  zxidappdemo.java
 * Copyright (c) 2014-2015 Synergetics (<EMAIL>), All Rights Reserved.
 * Author: <NAME> (<EMAIL>)
 * This is confidential unpublished proprietary source code of the author.
 * NO WARRANTY, not even implied warranties. Contains trade secrets.
 * Distribution prohibited unless authorized in writing.
 * Licensed under Apache License 2.0, see file COPYING.
 * $Id: zxidappdemo.java,v 1.4 2009-11-29 12:23:06 sampo Exp $
 * 20150211, changed to assume mod_auth_saml frontend --Sampo
 *
 * See also: zxid-java.pd, zxidwspdemo.java for server side
 * https://sp.personaldata.eu:8443/e2eTA/app-demo
 *
 * sudo apt-get install openjdk-6-jdk
 * cd syn-e2eta-connector-1.32-Linux-x86_64
 * javac -classpath /usr/share/tomcat6/lib/tomcat6-servlet-2.5-api-6.0.24.jar:. world.java
 * javac -classpath /usr/share/tomcat6/lib/servlet-api.jar:. world.java
 * javac -classpath /usr/share/tomcat6/lib/servlet-api.jar:../syn-e2eta-connector-1.22-Linux-x86_64/e2eta.jar world.java
 */

import e2eta.*;
import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.util.Enumeration;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class world extends HttpServlet {
    private static final long serialVersionUID = 1L;
    static e2eta.e2eta_conf cf;
    static String ptm;

    public static String ReadAll(String path) throws IOException {
        FileReader in = new FileReader(path);
        StringBuilder sb = new StringBuilder();
        char[] buf = new char[4096];
        int got = 0;
        do {
            sb.append(buf, 0, got);
            got = in.read(buf);
        } while (got >= 0);
	in.close();
        return sb.toString();
    }

    static {
	System.loadLibrary("e2etajni");
	try {
	    ptm = ReadAll("ptm-include.html");  // in /var/lib/tomcat6/, restart tomcat for update
	} catch(IOException e) {
	    System.err.print("File not found(ptm-include.html)\n");
	    System.err.print("Working Directory(" + System.getProperty("user.dir") + ")\n");
	}
    }

    public void doGet(HttpServletRequest req, HttpServletResponse res)
	throws ServletException, IOException
    {
	String fullURL = req.getRequestURI();
	String qs = req.getQueryString();
	if (qs != null)
	    fullURL += "?" + req.getQueryString();
	else
	    qs = "";
	System.err.print("========= Start World GET("+fullURL+")...\n");
	res.setContentType("text/html");
	ServletOutputStream out = res.getOutputStream();
	out.print("<title>World</title>\n");
	out.print("<link type=\"text/css\" rel=stylesheet href=\"/idpsel.css\">\n<body>");
	out.print("<h1>World</h1>\n");
	
	if (cf == null) {
	    System.err.print("Running conf\n");
	    String conf = getServletConfig().getInitParameter("E2ETAConf"); 
	    // CONFIG: You must have created /var/e2eta directory hierarchy with
	    // CONFIG: `e2etacot -dirs' and edited /var/e2eta/e2eta.conf to set the BURL
	    cf = e2etajni.new_conf_to_cf(conf);
	    System.err.print("dumpconf("+e2etajni.show_conf(cf)+").\n"); // Optional
	}
	if (ptm != null) {
	    out.print("<table align=right><tr><td>");
	    out.print("<a href=\"http://synergetics.be/\"><img src=\"synlogo_s.jpg\" height=67 border=0></a><br>");
	    out.print(ptm);
	    out.print("</td></tr></table>");
	}
	
 	out.print("<pre>RemoteUser("+req.getRemoteUser()+")\n");
 	out.print("RemoteUserHeader("+req.getHeader("REMOTE_USER")+")\n");
 	out.print("QueryString("+req.getQueryString()+")\n");
 	out.print("SAML_sesid("+req.getAttribute("SAML_sesid")+")\n");
 	out.print("SAML_affid("+req.getAttribute("SAML_affid")+")\n");
 	out.print("SAML_idpnid("+req.getAttribute("SAML_idpnid")+")\n");
 	out.print("SAML_fedusername("+req.getAttribute("SAML_fedusername")+")\n");
 	out.print("SAML_cn("+req.getAttribute("SAML_cn")+")\n");
 	out.print("SAML_lang("+req.getAttribute("SAML_lang")+")\n");
 	out.print("SAML_role("+req.getAttribute("SAML_role")+")\n");
 	out.print("SAML_o("+req.getAttribute("SAML_o")+")\n");
 	out.print("SAML_ou("+req.getAttribute("SAML_ou")+")\n");
	out.print("</pre><p>Done.\n");

	String ret;
	e2eta_ses ses = e2etajni.fetch_ses(cf, req.getAttribute("SAML_sesid").toString());
	
	out.print("<p>Output from PDS web service call rsrc=pds/flow/admin:<br>\n<textarea cols=80 rows=3>");
	ret = e2etajni.call(cf, ses, "urn:syn:pds:2015", null, null, "appid=CardiacFlow",
			    "{\"op\":\"read\",\"rsrc\":\"pds/flow/admin\"}");
        ret = e2etajni.extract_body(cf, ret);
        out.print(ret);
        out.print("</textarea>");

	out.print("<p>Output from PDS web service call rsrc=pds/flow/diseases:<br>\n<textarea cols=80 rows=3>");
	ret = e2etajni.call(cf, ses, "urn:syn:pds:2015", null, null, "appid=CardiacFlow",
			    "{\"op\":\"read\",\"rsrc\":\"pds/flow/diseases\"}");
        ret = e2etajni.extract_body(cf, ret);
        out.print(ret);
        out.print("</textarea>");

	out.print("<p>Output from PDS web service call rsrc=pds/flow/treatments:<br>\n<textarea cols=80 rows=3>");
	ret = e2etajni.call(cf, ses, "urn:syn:pds:2015", null, null, "appid=CardiacFlow",
			    "{\"op\":\"read\",\"rsrc\":\"pds/flow/treatments\"}");
        ret = e2etajni.extract_body(cf, ret);
        out.print(ret);
        out.print("</textarea>");

	out.print("<p>Output from PDS web service call rsrc=pds/flow/cardio/history:<br>\n<textarea cols=80 rows=3>");
	ret = e2etajni.call(cf, ses, "urn:syn:pds:2015", null, null, "appid=CardiacFlow",
			    "{\"op\":\"read\",\"rsrc\":\"pds/flow/cardio/history\"}");
        ret = e2etajni.extract_body(cf, ret);
        out.print(ret);
        out.print("</textarea>");

	System.err.print("^^^^^^^^^ DONE world GET("+fullURL+").\n");
    }
}

/* EOF */