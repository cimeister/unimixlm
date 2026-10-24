import React from 'react'
import PropTypes from 'prop-types'
import Cell from './Cell'
import grainImage from '../functions/grainImage'
import {findCell} from '../functions/cellData'
import {range, rangeLength} from '../functions/rangeLib'
import style from './Row.css'

export default class Row extends React.Component{
  render(){
    //breakable depending on padding...
    const start = this.props.xStart
    const y = this.props.y
    const size = this.props.size
    const cols = rangeLength(start,this.props.xEnd)
    const rowWidth = (size + 1)*cols
    
    return (
      <div 
        className='Row'
        style={{width: `${rowWidth}px`}}
      >
        {range(cols, start,1).map((i)=>{
          const cell = findCell(i,y,this.props.data) 
          return(
            <Cell 
              key={`${cell.x},${cell.y}`}
              x={cell.x}
              y={cell.y}
              size={size} 
              image={grainImage(size,cell.grains)}
              click={()=>this.props.click(cell)}  
            />
          )
        })
      }
      </div>
    )
  }
}

Row.propTypes = {
  size: PropTypes.number.isRequired,
  data: PropTypes.arrayOf(PropTypes.object),
  xStart: PropTypes.number.isRequired,
  xEnd: PropTypes.number.isRequired,
  y: PropTypes.number.isRequired,
  click: PropTypes.func.isRequired
}