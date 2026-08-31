/* eslint-disable no-console */
import Node from './node.js';
import { appendRectText } from './util.js';


export default class TableNode extends Node {
  constructor(data, children) {
    super(data, children);

    this.textPad = 10;
  }

  _render(g) {
    if (this.data.length === 0 || this.data[0].length === 0) {
      return;
    }

    const tbl = this.renderCells(g);
    const size = this.layoutCells(tbl);

    this.width = size.width;
    this.height = size.height;

    this.linkX = Math.round(this.width / 2);
    this.linkY = Math.round(this.height / 2);
  }

  renderCells(g) {
    const tbl = [];

    this.data.forEach((row) => {
      const tblRow = [];

      row.forEach((col,i) => {
        // eslint-disable-next-line no-console
        // console.log(row)
        if(i==1){
          
        
        const colG = g.append('g');
        const bbox = appendRectText(colG, 0, 0, col, this.textPad);

        tblRow.push({
          g: colG,
          bbox: bbox
        });
      }
      });

      tbl.push(tblRow);
    });

    return tbl;
  }

  layoutCells(tbl) {
    const maxW = this.calcMaxWidths(tbl);
    const maxH = this.calcMaxHeights(tbl);

    var x = 0;
    var y = 0;

    tbl.forEach((row, rowI) => {
      x = 0;

      row.forEach((col, colI) => {
        col.g.attr('transform', `translate(${x},${y})`);
        col.g.select('rect')
          .attr('width', maxW[colI])
          .attr('height', maxH[rowI]);

        x += maxW[colI];
      });

      y += maxH[rowI];
    });

    return { width: x, height: y };
  }

  calcMaxWidths(tbl) {
    const maxW = [];
    var colI, rowI;

    for (colI = 0; colI < tbl[0].length; colI++) {
      var w = 0;

      for (rowI = 0; rowI < tbl.length; rowI++) {
        w = Math.max(w, tbl[rowI][colI].bbox.width);
      }
      if(w<60){
        maxW.push(70);
      }else{
        maxW.push(w)
      }
      
    }

    return maxW;
  }

  calcMaxHeights(tbl) {
    const maxH = [];

    tbl.forEach((row) => {
      var h = 0;

      row.forEach((col) => {
        h = Math.max(h, col.bbox.height);
      });

      maxH.push(h+10);
    });

    return maxH;
  }
}