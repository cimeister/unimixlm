import React from "react"

const Container = props => {
  const { title, desc, classes } = props
  return (
    <section className={classes}>
      <div className="container">
        <div className="row d-flex justify-content-center">
          <div className="menu-content pb-70 col-lg-8">
            <div className="title text-center">
              <h1 className="mb-10">{title}</h1>
              <p>{desc}</p>
            </div>
          </div>
        </div>
        {props.children}
      </div>
    </section>
  )
}

export default Container