import React from 'react'
import {withRouter, Route, Switch, Link} from 'react-router-dom'
import {connect} from 'react-redux'
import {MainForm} from './main-form'
import {OptionsBar} from './options-bar'

/**
 * COMPONENT
 */
export const ChosenCardMini = props => {
  return (
    <Link to={`/savedTrip/${props.chosenLocation.id}`}>
      <div className="trip-item">
        <p>
          {props.chosenLocation.name},{props.chosenLocation.state}
        </p>
      </div>
    </Link>
  )
}