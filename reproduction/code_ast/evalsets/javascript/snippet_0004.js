import React from "react"
// import { Link } from "gatsby"
import Layout from "../components/layout"
import SEO from "../components/seo"
import { navigate } from "@reach/router"
import { graphql } from "gatsby"
import { format } from "date-fns"
import Links from "../components/Links"
import FullCalendar from "@fullcalendar/react"
import dayGridPlugin from "@fullcalendar/daygrid"
import timeGridPlugin from "@fullcalendar/timegrid"
import interactionPlugin from "@fullcalendar/interaction" // needed for dayClick
import "@fullcalendar/core/main.css"
import "@fullcalendar/daygrid/main.css"
import "@fullcalendar/timegrid/main.css"

const IndexPage = ({ data }) => {
  const postArray = []
  data.allMarkdownRemark.edges.map(post => {
    postArray.push({
      title: post.node.frontmatter.description,
      date: post.node.frontmatter.date,
    })
    return postArray
  })
  return (
    <Layout>
      <SEO title="Home" />
      <FullCalendar
        className="demo-app-calendar"
        defaultView="dayGridMonth"
        header={{
          left: "prev,next today",
          center: "title",
          right: "dayGridMonth,timeGridWeek,timeGridDay,listWeek",
        }}
        events={postArray}
        eventClick={e => {
          navigate(`../${format(e.event.start, "yyyy-MM-dd")}`)
        }}
        plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
      />
      {/* <Link to="/page-2/">Go to page 2</Link> */}
      <Links posts={data.allMarkdownRemark.edges} />
    </Layout>
  )
}

export const pageQuery = graphql`
  query IndexQuery {
    allMarkdownRemark {
      edges {
        node {
          id
          frontmatter {
            path
            title
            date
            description
          }
        }
      }
    }
  }
`

export default IndexPage