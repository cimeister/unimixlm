import request from '@/utils/request'
import Urls from './urls'

export function loginByUsername(email, password) {
  const data = {
    email,
    password
  }
  return request({
    url: Urls.login,
    method: 'post',
    data
  })
}

export function logout() {
  return request({
    url: '/login/logout',
    method: 'post'
  })
}

export function getUserInfo() {
  return request({
    url: Urls.current,
    method: 'get'
  })
}

// 注册
export function register(data) {
  return request({
    url: Urls.register,
    method: 'post',
    data
  })
}

// 获取验证码
export function getCaptch(type) {
  const _now = new Date().getTime()
  return request({
    url: `${Urls.captcha}?type=${type}&t=${_now}`,
    method: 'get'
  })
}