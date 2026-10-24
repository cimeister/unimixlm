'use strict'

const server = require('./lib/server')

const port = 3000

server.listen(port, () => {
  console.log('Server up and running on port ' + port)
})