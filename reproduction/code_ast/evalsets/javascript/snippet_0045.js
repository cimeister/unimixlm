import { racket } from './racket/racket'
import { kryproflow } from './kryptoflow/kryptoflow'
import { vapr } from './VAPr/vapr'
import { mhcpreds } from './mhcPreds/mhcPreds'

export let projects = [
  {
    ...racket,
    id: 1
  },
  {
    ...kryproflow,
    id: 2
  },
  {
    ...vapr,
    id: 3
  },
  {
    ...mhcpreds,
    id: 4
  },

]