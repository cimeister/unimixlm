import React from "react";
import developer from "../assets/images/roshan_bajracharya.jpg"; 
 
function About(){
    return (
        <section id="about-me" className="about-me spacer">
      <div className="container">
        <div className="row">
          <div className="column-half">
            <div className="about-content">
              <h3>About Me</h3>
              <h2><NAME></h2>
              <p>Hi I am <NAME>. I am currently living in Adelaide, South Australia. Basically I am from
                Nepal. I have keen interest in Web Development and I have worked in this field back in Nepal.</p>
              <p>My main motivation to join this course is to update myself and learn the latest trends in the Web
                Development Sector.</p>
              <a href="/" className="btn">Know more</a>
            </div>
          </div>
          <div className="column-onethird">
            <figure><img src={developer} alt="<NAME>" /></figure>
          </div>
        </div>
      </div>
    </section>
    );
  }
 
export default About;