import React from 'react'
import { Grid, Header, Segment, Icon } from 'semantic-ui-react'

const tamanho = {
    width: 275,
    heigth: 175
}

const Category = props =>{

    return(
         <Grid.Column>
            <Segment circular style={tamanho}>
                <Header>
                <Header.Subheader>                           
                    <Icon name={props.icone} />
                    </Header.Subheader>
                    {props.titulo}            
                </Header>
           </Segment> 
         </Grid.Column>
    )
}

export default Category