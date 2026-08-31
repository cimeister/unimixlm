'use strict'

const Promise = require('bluebird')
const _ = require('lodash')
const moment = require('moment')
const consts = require('../config/consts')
const config = require('../config/config')
const lang = require('../config/lang.vi.js')
const utils = require('../utils/utils')
const CacheHandler = require('../logic/cache-handler')
const CustomerMemory = require('../logic/customer-memory')
const geoip = require('geoip-lite')
const TextService = require('../services/text-service')

module.exports = function(Home) {

  Home.beforeRemote('*', (ctx, config, next) => {
    const token = _.get(ctx, 'args.options.accessToken.role')
    const isAdmin = _.get(ctx, 'args.options.authorizedRoles.admin')
    if ((!token || !isAdmin)) {
      const filterFields = _.get(ctx, 'args.filter.fields', [])
      if (_.isArray(filterFields)) {
        const fields = {}
        _.forEach(filterFields, field => {
          fields[field] = true
        })
        _.set(ctx, 'args.filter.fields', fields)
      }

      _.set(ctx, 'args.filter.fields.router', false)
      _.set(ctx, 'args.filter.fields.query', false)
    }
    next()
  })

  Home.remoteMethod('getHome', {
    accepts: [
      {arg: 'req', type: 'object', http: { source: 'req' }},
      {arg: 'versionCode', type: 'number'},
      {arg: 'versionJs', type: 'number'},
      {arg: 'getMenu', type: 'number'},
      {arg: 'filter', type: 'object', default: {}}
    ],
    returns: {type: 'object', root: true},
    http: {verb: 'get'},
    description: 'Get home'
  })

  Home.getHome = (req, versionCode, versionJs, getMenu, filter, cb) => {
    if (!req || !req.username) {
      return cb({ statusCode: consts.CODE.KICK })
    }

    const deviceDtId = req.query.dtId || (filter.dtId || req.dtId)
    const ip = utils.getIp(req)
    const geo = geoip.lookup(ip) || {}

    const operators = {
      homeList: Home.find({
        where: {
          activated: true
        },
        fields: { router: false, query: false, createdAt: false, updatedAt: false, activated: false },
        cacheTime: config.CACHE_TIME
      }),
      hasNotUnRead: CustomerMemory.getHasNotUnRead(req.username)
    }

    if (getMenu) {
      operators.menuList = Home.app.models.Menu.find({
        where: {
          activated: true
        },
        fields: {createdAt: false, updatedAt: false, activated: false},
        cacheTime: config.CACHE_TIME
      })
    }

    versionCode = versionCode || (filter.versionCode || 0)

    console.log('DEBUG_UPDATE', versionCode)
    if (versionCode) {
      operators.config = Home.app.models.Config.getConfigByRequest(req, ['version'], filter)
    }

    Promise.props(operators)
      .then(data => {
        let homeList = _.sortBy(data.homeList, 'rank')

        // todo fix for LG, tizen DRM
        if ([consts.PLATFORM.LG, consts.PLATFORM.TIZEN].indexOf(req.platform) >= 0) {
          homeList = _.filter(homeList, item => item.name != 'PHIM HOT' && item.name != 'PHIM YÊU THÍCH')
        }

        const result = {
          data: homeList,
          config: Home.config,
          drm: {},
          username: req.username !== 'GUEST' ? req.username : req.guestId,
          uid: req.username !== 'GUEST' ? Number(req.username.substring(2)).toString(16).toUpperCase() : req.guestId,
          hasUnRead: data.hasNotUnRead ? 0 : 1,
          inReview: (utils.inReview({ dtId: deviceDtId, platform: req.platform }) || geo.country != 'VN') ? 1 : 0
        }

        const isWebPlatform = req.platform == consts.PLATFORM.WEB
          || req.platform == consts.PLATFORM.LG
          || req.platform == consts.PLATFORM.TIZEN
        result.config.pingUrl = isWebPlatform ? config.PING_URL_WEB : config.PING_URL_APP

        if (data.menuList) {
          result.menu = _.sortBy(data.menuList, 'rank')
        }

        const version = (data.config && data.config.version) || {}

        if (
          versionCode
          && version.version
          && versionCode < version.version
        ) {
          result.newVersion = {
            version: version.version,
            message: lang.updateMessage(moment(version.version, 'YYYYMMDD').format('DD/MM/YYYY')),
            updateUrl: Number(filter.needFullScreen) === 0
              ? version.updateUrl.replace('.apk', '-notFullscreen.apk') : version.updateUrl,
            forceUpdate: version.forceUpdate ? 1 : 0,
            type: version.type || 1,
            fileSize: version.fileSize || 0
          }
        }

        // result.config.ads = {
        //   adsId: '258770',
        //   maxDay: 20,
        //   maxSession: 5,
        //   delayTime: 600,
        //   delayContent: 0,
        //   midrollTime: [600]
        // }

        cb(null, result)

        publishGetHome(req)
      })
      .catch(e => {
        console.error('getHome err', e.stack || e)
        cb(e)
      })
  }

  Home.config = {
    homeRefreshTime: config.HOME_REFRESH_TIME,
    channelRefreshTime: config.CHANNEL_REFRESH_TIME,
    menuRefreshTime: config.MENU_REFRESH_TIME,
    streamTokenExpire: config.VIETTEL_SECURE_CHANNEL.ttl,
    marqueeRunTime: config.MARQUEE_RUN_TIME,
    marqueeInterval: config.MARQUEE_INTERVAL,
    pingInterval: config.PING_INTERVAL,
    notifyUrl: config.NOTIFY_URL,
    youtubeProcessUrl: config.YOUTUBE_PROCESSING_URL,
    drmUrls: config.DRM_URLS,
    baseImageUrl: config.BASE_IMAGE_URL,
    screenPingInterval: config.SCREEN_PING_INTERVAL,
    witService: config.witService
  }

  function publishGetHome(req) {
    return Home.app.get('rabbit').publish({
      channel: consts.RABBIT_CHANNEL.HOME,
      data: {
        time: moment().unix(),
        username: req.username,
        deviceId: req.deviceId,
        deviceType: req.deviceType,
        dtId: req.dtId,
        spId: req.spId,
        ip: utils.getIp(req)
      }
    })
  }

  Home.remoteMethod('getMenu', {
    accepts: [
      {arg: 'req', type: 'object', http: { source: 'req' }},
      {arg: 'id', type: 'string'},
      {arg: 'type', type: 'number', default: 1},
      {arg: 'parentId', type: 'string'}
    ],
    returns: {type: 'object', root: true},
    http: {verb: 'get'},
    description: 'Get menu cates'
  })

  Home.getMenu = (req, id, type, parentId, cb) => {
    let model

    switch (type) {
      case consts.MENU_TYPE.MOVIE:
        model = 'MovieCatalog'
        break

      case consts.MENU_TYPE.CLIP:
        model = 'ClipCatalog'
        break

      case consts.MENU_TYPE.CHANNEL:
        model = 'ChannelCatalog'
        break
    }

    if (!model)
      return cb(null, {data: []})

    const where = {
      activated: true
    }

    if (parentId) where.parentId = parentId

    Home.app.models[model].find({
      where: where,
      fields: ['id', 'name', 'rank'],
      cacheTime: config.CACHE_TIME
    })
      .then(list => {

        for (let i = 0; i < list.length; i++) {
          list[i].id = 'menu_'+list[i].id
          list[i].type = type
          list[i].hasLoadMore = true
          list[i].hasViewAll = true
          list[i].page = 1
          list[i].limit = req.deviceType == consts.DEVICE_TYPE.TV
            ? consts.DEFAULT_TV_LIMIT
            : consts.DEFAULT_MOBILE_LIMIT
        }

        cb(null, {data: _.sortBy(list, 'rank')})
      })
      .catch(e => {
        console.error('getMenu err', e.stack || e)
        cb(null, {data: []})
      })
  }

  Home.getMenuData = (req, type, query, page, cb) => {
    let model
    let catalogField

    switch (type) {
      case consts.MENU_TYPE.MOVIE:
        model = 'Movie'
        catalogField = 'movieCatalogIds'
        break

      case consts.MENU_TYPE.CLIP:
        model = 'Clip'
        catalogField = 'clipCatalogIds'
        break

      case consts.MENU_TYPE.CHANNEL:
        model = 'Channel'
        catalogField = 'channelCatalogIds'
        break
    }

    if (!query.id || !model) {
      return utils.invokeCallback(cb, null, [])
    }

    const filter = {
      where: {
        activated: true,
        [catalogField]: query.id.split('_').pop()
      },
      order: 'updatedAt DESC'
    }
    filter.limit = req.deviceType == consts.DEVICE_TYPE.TV
      ? consts.DEFAULT_TV_LIMIT
      : consts.DEFAULT_MOBILE_LIMIT
    filter.skip = ((page || 1) - 1) * filter.limit
    filter.fields = 'default'
    filter.cacheTime = config.CACHE_TIME

    // todo fix for LG, tizen DRM
    if ([consts.PLATFORM.LG, consts.PLATFORM.TIZEN].indexOf(req.platform) >= 0) {
      filter.where.createdAt = { lte: new Date('2018-10-01') }
    }

    // hide unlicensed content
    if (config.DISABLE_UNLICENSED_CONTENT && utils.isFromStore(req)) {
      filter.where = filter.where || {}
      filter.where.activatedStore = true
    }

    const promise = Home.app.models[model].find(filter)
      .then(result => {
        // _.each(result, (item, i) => {
        //   if (item.source && item.source.indexOf('youtube')) {
        //     result[i].youtubeId = item.source.split('=')[1]
        //   }
        //   item.source = undefined
        // })

        if (model == 'Movie') {
          _.each(result, (movie, i) => {
            movie.providerLogo = _.get(consts, `PACKAGE.${movie.packageCode}.logo`) || null
          })
        }

        // kênh địa phương chỉ lấy lân cận
        // if (model == 'Channel' && query.id.split('_').pop() == '5a8f8ecd4d6ca32966ad4fdf') {
        //   const ip = utils.getIp(req)
        //   const geo = geoip.lookup(ip) || {}
        //   const city = (geo.city || '').toLowerCase()
        //
        //   console.log('KENH DIA PHUONG', city)
        //
        //   result = _.filter(result, (item) => {
        //     return !!(item.keywords && item.keywords.length && item.keywords.indexOf(city) != -1)
        //   })
        // }

        return utils.invokeCallback(cb, null, result || [])
      })
      .catch(e => {
        console.error('get menu list err', e.stack || e)
        return utils.invokeCallback(cb, null, [])
      })

    if (!cb) return promise
  }

  Home.remoteMethod('getListQueue', {
    accepts: [
      {arg: 'req', type: 'object', http: { source: 'req' }},
      {arg: 'queue', type: 'object', required: true},
    ],
    returns: {type: 'array', root: true},
    http: {verb: 'get'},
    description: 'Get home list by ID queue'
  })

  Home.getListQueue = (req, queue, cb) => {
    if (!queue || !Object.keys(queue).length) {
      return cb(null, [])
    }

    const operators = {}

    _.forEach(queue, (command, id) => {
      if (command && command.type && command.filter && command.filter.id) {
        operators[id] = (command.filter.id.indexOf('menu_') === 0)
          ? Home.getMenuData(req, command.type, command.filter, command.page || 1)
          : Home.getHomeData(req, command.filter, command.page || 1)
      } else {
        operators[id] = []
      }
    })

    Promise.props(operators)
      .then(result => {
        cb(null, result)
      })
      .catch(e => {
        e && console.error('getListQueue error: ', e.stack || e)
        cb(e)
      })
  }

  Home.remoteMethod('getList', {
    accepts: [
      {arg: 'req', type: 'object', http: { source: 'req' }},
      {arg: 'type', type: 'number', default: 1},
      {arg: 'filter', type: 'object'},
      {arg: 'page', type: 'number', default: 1}
    ],
    returns: {type: 'array', root: true},
    http: {verb: 'get'},
    description: 'Get home list by ID'
  })

  Home.getList = (req, type, query, page, cb) => {
    if (!query.id) {
      return utils.invokeCallback(cb, null, [])
    }

    if (query.id.indexOf('menu_') == 0) {
      Home.getMenuData(req, type, query, page, cb)
    } else {
      Home.getHomeData(req, query, page, cb)
    }
  }

  Home.getHomeData = (req, query, page, cb) => {

    const promise = Home.findById(query.id, {cacheTime: config.CACHE_TIME})
      .then(homeItem => {
        if (!homeItem || !homeItem.router)
          return Promise.reject(new Promise.CancellationError())

        const [DataModel, method] = homeItem.router.split('.')
        const filter = homeItem.query || {}
        filter.limit = homeItem.limit || consts.DEFAULT_TV_LIMIT
        filter.skip = ((page || 1) - 1) * filter.limit
        filter.fields = 'default'

        // todo fix for LG, tizen
        if ([consts.PLATFORM.LG, consts.PLATFORM.TIZEN].indexOf(req.platform) >= 0) {
          if (DataModel === 'Slide') {
            filter.where = { type: 1 }
          } else if (DataModel === 'Movie') {
            filter.where.createdAt = { lte: new Date('2018-10-01') }
          }
        }

        // hide unlicensed content
        if (config.DISABLE_UNLICENSED_CONTENT && utils.isFromStore(req)) {
          filter.where = filter.where || {}
          filter.where.activatedStore = true
        }

        if (homeItem.isPersonal || homeItem.type == consts.MENU_TYPE.CHANNEL) {
          return Home.app.models[DataModel][method](req, filter)
        } else {
          filter.cacheTime = config.CACHE_TIME
          return Home.app.models[DataModel][method](filter)
        }
      })
      .then(result => {
        // _.each(result, (item, i) => {
        //   if (item.source && item.source.indexOf('youtube')) {
        //     result[i].youtubeId = item.source.split('=')[1]
        //   }
        //   item.source = undefined
        // })
        return utils.invokeCallback(cb, null, result || [])
      })
      .catch(Promise.CancellationError, () => {
        console.error('CANCEL HOME')
        return utils.invokeCallback(cb, null, [])
      })
      .catch(e => {
        console.error('home get list err', e.stack || e)
        return utils.invokeCallback(cb, null, [])
      })

    if (!cb) return promise
  }

  Home.remoteMethod('suggest', {
    accepts: {arg: 'query', type: 'string', required: true},
    returns: {type: 'array', root: true},
    http: {verb: 'get'},
    description: 'Search suggestion'
  })

  Home.suggest = (query, cb) => {
    if (query.length < 2)
      return cb(null, [])

    query = utils.locDauTV(query).toLowerCase()

    const elastic = Home.app.get('elastic')
    const fields = ['name', 'nameKoDau', 'nameCompress', 'actors']

    elastic.search({
      index: 'movie,channel,clip',
      body: {
        _source: ['name'],
        query: {
          multi_match: {
            query: query,
            fields: fields
          }
        },
        size: 6
      }
    })
      .then(rs => {
        const result = rs.hits
          ? _(rs.hits.hits).map(item => _.get(item, '_source.name')).value()
          : []
        cb(null, result)
      })
      .catch(e => {
        console.error('search suggestion error', e.stack || e)
        cb(null, [])
      })
  }

  Home.remoteMethod('search', {
    accepts: [
      {arg: 'req', type: 'object', http: { source: 'req' }},
      {arg: 'query', type: 'string', required: true}
    ],
    returns: {type: 'array', root: true},
    http: {verb: 'get'},
    description: 'Search'
  })

  Home.search = (req, query, cb) => {
    if (query.length < 2)
      return cb(null, [])

    query = query.toLowerCase()

    const elastic = Home.app.get('elastic')
    let movieIds, channelIds, clipIds
    const fields = ['name', 'nameKoDau', 'nameCompress', 'actors']

    elastic.search({
      index: 'movie,channel,clip',
      body: {
        _source: false,
        query: {
          multi_match: {
            query: query,
            type: query.split(' ').length >= 3 ? 'phrase' : 'best_fields',
            fields: fields
          }
        },
        size: 20
      }
    })
      .then(rs => {
        if (!rs || !rs.hits) {
          return Promise.reject(new Promise.CancellationError())
        }

        const hits = _(rs.hits.hits)
        movieIds = hits
          .filter(item => item._index == 'movie')
          .map(item => item._id.toString())
          .value()

        channelIds = hits
          .filter(item => item._index == 'channel')
          .map(item => item._id.toString())
          .value()

        clipIds = hits
          .filter(item => item._index == 'clip')
          .map(item => item._id.toString())
          .value()

        const listMovies = movieIds && movieIds.length
          ? CacheHandler.getListCache('Movie', { id: movieIds.slice() }, { fields: 'default' })
          : Promise.resolve([])

        const listChannels = channelIds && channelIds.length
          ? CacheHandler.getListCache('Channel', { id: channelIds.slice() }, { fields: 'default' })
          : Promise.resolve([])

        const listClips = clipIds && clipIds.length
          ? CacheHandler.getListCache('Clip', { id: clipIds.slice() }, { fields: 'default' })
          : Promise.resolve([])

        return Promise.all([
          listMovies,
          listChannels,
          listClips
        ])
      })
      .spread((movies, channels, clips) => {
        const result = []
        const filter = (item) => {
          return true // todo filter activated
        }
        if (channels && channels.length) {
          result.push({
            type: consts.MEDIA_TYPE.CHANNEL,
            title: lang.channel,
            data: _(channels)
              .filter(item => filter(item))
              .sortBy(item => channelIds.indexOf(item.id.toString()))
              .value()
          })
        }

        if (clips && clips.length) {
          result.push({
            type: consts.MEDIA_TYPE.CLIP,
            title: lang.clip,
            data: _(clips)
              .filter(item => filter(item))
              .sortBy(item => clipIds.indexOf(item.id.toString()))
              .value()
          })
        }

        if (movies && movies.length) {
          result.push({
            type: consts.MEDIA_TYPE.MOVIE,
            title: lang.movie,
            data: _(movies)
              .filter(item => filter(item))
              .sortBy(item => movieIds.indexOf(item.id.toString()))
              .value()
          })
        }

        cb(null, result)
      })
      .catch(Promise.CancellationError, e => {
        cb(null, [])
      })
      .catch(e => {
        console.error('search error', e.stack || e)
        cb(null, [])
      })
  }

  const screenMap = {
    movie: consts.SCREEN.MOVIE_DETAIL,
    clip: consts.SCREEN.CLIP_DETAIL,
    channel: consts.SCREEN.CHANNEL_DETAIL
  }

  const mapIndexToType = {
    'movie': consts.MENU_TYPE.MOVIE,
    'clip': consts.MENU_TYPE.CLIP,
    'channel': consts.MENU_TYPE.CHANNEL
  }

  Home.remoteMethod('command', {
    accepts: {arg: 'query', type: 'string', required: true},
    returns: {type: 'array', root: true},
    http: {verb: 'get'},
    description: 'Command'
  })

  Home.command = (query, cb) => {
    if (query.length < 2)
      return cb(null, [])

    query = query.toLowerCase().trim()

    const elastic = Home.app.get('elastic')
    const fields = ['name', 'nameKoDau', 'nameCompress', 'actors']

    // process to TV channel
    if (query.indexOf('tv') >= 0) {
      const tmp = query.split('tv')
      if (tmp.length === 2)
      query = tmp[0] + 'tv ' + tmp[1]
    }

    elastic.search({
      index: 'channel,movie,clip',
      body: {
        _source: ['name'],
        query: {
          multi_match: {
            query: query,
            type: 'phrase', //query.split(' ').length >= 3 ?  : 'best_fields',
            fields: fields
          }
        },
        size: 20
      }
    })
      .then(rs => {
        if (!rs || !rs.hits || !rs.hits.hits || !rs.hits.hits.length) {
          return Promise.reject(new Promise.CancellationError())
        }

        rs.hits.hits = _(rs.hits.hits)
          .map(hit => {
            if (hit._index === 'channel') {
              hit._score *= 3
            }

            return hit
          })
          .orderBy(['_score'], ['desc'])
          .value()

        console.log('DEBUG_COMMAND', query, rs.hits.hits)

        if (rs.hits.hits[0]._score >= consts.SEARCH_SCORE_THRESHOLD) {
          const item = rs.hits.hits[0]
          let screen = screenMap[item._index] || consts.SCREEN.UNKNOWN
          cb(null, { query, target: { screen, id: item._id } })
        } else {
          const keywords = rs.hits
            ? _(rs.hits.hits).map(item => ({
              name: _.get(item, '_source.name'),
              id: _.get(item, '_id'),
              type: mapIndexToType[_.get(item, '_index')]
            })).value()
            : []

          cb(null, { query, target: {
            screen: consts.SCREEN.SEARCH,
            keywords: _(rs.hits.hits).map(item => _.get(item, '_source.name')).value(),
            result: keywords
          } })
        }
      })
      .catch(Promise.CancellationError, e => {
        cb(null, { target: { screen: consts.SCREEN.UNKNOWN }, query })
      })
      .catch(e => {
        console.error('search error', e.stack || e)
        cb(null, { target: { screen: consts.SCREEN.UNKNOWN }, query })
      })
  }

  Home.afterRemote('find', (ctx, user, next) => {
    const token = _.get(ctx, 'args.options.accessToken')
    if(!ctx.result || !token || !_.get(ctx, 'args.filter.getConfig'))
      return next()

    const now = utils.momentRound(moment(), moment.duration(config.CACHE_TIME, 'seconds'), 'ceil').toDate()

    const notifyOnline =
      ctx.req.deviceType == consts.DEVICE_TYPE.TV
        ? Promise.resolve(0)
        : Home.app.models.Notification.findOne(
        {
          where: {
            activated: true,
            type: consts.NOTIFY_TYPE.ONLINE,
            showTime: { lte: now },
            expireAt: { gte: now }
          },
          order: 'showTime DESC',
          cacheTime: config.CACHE_TIME
        })

    const notifyOffline =
      ctx.req.deviceType == consts.DEVICE_TYPE.TV
        ? Promise.resolve(0)
        : Home.app.models.Notification.findOne(
        {
          where: {
            activated: true,
            type: consts.NOTIFY_TYPE.OFFLINE,
            showTime: { gte: now },
            expireAt: { gte: now }
          },
          order: 'showTime DESC',
          cacheTime: config.CACHE_TIME
        })

    Promise.all([
      Home.app.models.Config
        .findOne({
          where: {
            dtId: ctx.req.dtId,
            deviceType: ctx.req.deviceType
          },
          fields: ['config', 'version', 'drm'],
          cacheTime: config.CACHE_TIME
        }),
      notifyOnline,
      notifyOffline
    ])
      .spread((row, notifyOnline, notifyOffline) => {
        const configData = row.config
        const version = row.version

        if (row.drm)
          row.drm.drmSecret = undefined

        const clientVersionCode = Number(_.get(ctx, 'args.filter.versionCode'))

        ctx.result = {
          data: _(ctx.result)
            .filter(item => [consts.MEDIA_TYPE.CLIP, consts.MEDIA_TYPE.PROGRAM].indexOf(item.type) == -1)
            .sortBy('rank')
            .value(),
          config: _.merge({ streamTokenExpire: config.VIETTEL_SECURE_CHANNEL.ttl }, configData),
          drm: row.drm,
          username: ctx.req.username,
          notifyOnline: notifyOnline ? moment(notifyOnline.createdAt).unix() : 0,
          notifyOffline: notifyOffline ? moment(notifyOffline.createdAt).unix() : 0
        }

        if (
          version.version
          && clientVersionCode
          && clientVersionCode < version.version
        ) {
          ctx.result.newVersion = {
            message: lang.updateMessage(moment(version.version, 'YYYYMMDD').format('DD/MM/YYYY')),
            updateUrl: version.updateUrl,
            forceUpdate: version.forceUpdate ? 1 : 0,
            type: version.type || 1,
            fileSize: version.fileSize || 0
          }

          // 'Hệ thống đang bảo trì và sẽ được mở lại sau vài ngày tới. Để biết thêm chi tiết, Quý Khách vui lòng liên hệ số điện thoại: 1900585868'
        }

        next()
      })
  })

  // Home.remoteMethod('elasticMovie', {
  //   returns: {type: 'array', root: true},
  //   http: {verb: 'get'},
  //   description: 'Elastic movie'
  // })
  //
  // Home.elasticMovie = (cb) => {
  //   let count = 0
  //   Home.app.models.Movie.find({})
  //     .each((movie, i) => {
  //       count++
  //       return Promise.delay(i).then(() => movie.save())
  //     })
  //     .then(() => {
  //       console.log('DONE elastic movie, count: ', count)
  //       cb(null, {})
  //     })
  //     .catch(e => e && console.error('elastic channel err: ', e.stack || e) && cb(null, {}))
  // }
  //
  // Home.remoteMethod('elasticClip', {
  //   returns: {type: 'array', root: true},
  //   http: {verb: 'get'},
  //   description: 'Elastic clip'
  // })
  //
  // Home.elasticClip = (cb) => {
  //   let count = 0
  //   Home.app.models.Clip.find({})
  //     .each((movie, i) => {
  //       count++
  //       return Promise.delay(i).then(() => movie.save())
  //     })
  //     .then(() => {
  //       console.log('DONE elastic clip, count: ', count)
  //       cb(null, {})
  //     })
  //     .catch(e => e && console.error('elastic clip err: ', e.stack || e) && cb(null, {}))
  // }
  //
  // Home.remoteMethod('elasticChannel', {
  //   returns: {type: 'array', root: true},
  //   http: {verb: 'get'},
  //   description: 'Elastic channel'
  // })
  //
  // Home.elasticChannel = (cb) => {
  //   let count = 0
  //   Home.app.models.Channel.find({})
  //     .each(channel => {
  //       count++
  //       return channel.save()
  //     })
  //     .then(() => {
  //       console.log('DONE elastic channel, count: ', count)
  //       cb(null, {})
  //     })
  //     .catch(e => e && console.error('elastic channel err: ',  e.stack || e) && cb(null, {}))
  // }

}