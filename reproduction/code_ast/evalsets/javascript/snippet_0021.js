import React from 'react'
import PropTypes from 'prop-types'
import IconButton from '@material-ui/core/IconButton'
import EditIcon from '@material-ui/icons/Edit'

const EditButton = ({ editAction }) => (
  <IconButton aria-label="Edit" onClick={() => editAction()}>
    <EditIcon/>
  </IconButton>
)

EditButton.propTypes = {
  editAction: PropTypes.func
}

export default EditButton