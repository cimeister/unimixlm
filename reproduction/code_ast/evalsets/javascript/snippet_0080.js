import React from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Card from "react-bootstrap/Card";
import { Link } from "react-router-dom";
import school from "../school.svg";
import sunny from "../sunny.svg";
import hundredsChart from "../hundreds-chart.svg";
import speaker from "../loudspeaker.svg";
import poem from "../poem.svg";
import calendar from "../calendar.svg";
import blocks from "../blocks.svg";

const Home = () => (
  <div className="Home">
    <Row>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={speaker}
                alt="Message icon"
                style={{
                  width: "24px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Morning Message
            </Card.Title>
            <Card.Text>Let's start our day with a morning message!</Card.Text>
            <Link className="btn btn-success" to="/morning/message">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={calendar}
                alt="Calendar icon"
                style={{
                  width: "24px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Daily Calendar
            </Card.Title>
            <Card.Text>Let's check out our daily calendar!</Card.Text>
            <Link className="btn btn-success" to="/calendar">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={sunny}
                alt="Sunny icon"
                style={{
                  width: "25px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Weather Graph
            </Card.Title>
            <Card.Text>Let's keep track of the weather!</Card.Text>
            <Link className="btn btn-success" to="/weather-graph">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
    </Row>
    <Row>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={school}
                alt="Calculator icon"
                style={{
                  width: "22px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Days in School
            </Card.Title>
            <Card.Text>
              Let's count how many days we've been in school!
            </Card.Text>
            <Link className="btn btn-success" to="/days-in-school">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={hundredsChart}
                alt="Chart icon"
                style={{
                  width: "25px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Hundreds Chart
            </Card.Title>
            <Card.Text>
              Let's reveal who has stolen numbers from the hundreds chart!
            </Card.Text>
            <Link className="btn btn-success" to="/hundreds-chart">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={poem}
                alt="Poem icon"
                style={{
                  width: "24px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Morning Poem
            </Card.Title>
            <Card.Text>Let's brighten our day with a poem!</Card.Text>
            <Link className="btn btn-success" to="/morning/poem">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
    </Row>
    <Row>
      <Col>
        <Card style={{ width: "18rem" }}>
          <Card.Body>
            <Card.Title>
              <img
                src={blocks}
                alt="Blocks icon"
                style={{
                  width: "26px",
                  height: "auto",
                  marginRight: "5px",
                  verticalAlign: "bottom"
                }}
              />
              Making Words
            </Card.Title>
            <Card.Text>
              Let's make some words with consonants, vowels and digraphs!
            </Card.Text>
            <Link className="btn btn-success" to="/making-words">
              Play
            </Link>
          </Card.Body>
        </Card>
      </Col>
    </Row>
  </div>
);

export default Home;