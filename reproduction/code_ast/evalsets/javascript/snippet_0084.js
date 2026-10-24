import store from '../store';
import Util from './util';
import axios from 'axios';
import _ from 'lodash';
import config from '../../build/config.js';

function install (Vue) {
    /**
     * 关闭当前页面，清除当前页的keepalive缓存，并跳转到新页面，本方法以尽量激活新页面为目的，如：
     * 1，如果新页面在缓存中，只是激活缓存，触发新页面的activated事件
     * 2，如果新页面没在缓存中，重新打开新页面，触发新页面的created事件
     * 3，新页面如果已经存在打开的多个，那么新页面重新创建一个新的，触发新页面的created事件
     * @param location 路由信息
     */
    Vue.prototype.$replacePageAndActiveRouter = function (location) {
        // console.log('this.$vnode.key',this.$vnode.key)
        this.$store.commit('removeTag', this.$vnode.key);
        this.$router.replace(location);
    };
    /**
     * 关闭当前页面，清除当前页的keepalive缓存，并跳转到新页面，本方法以尽量激活新页面为目的，如：
     * 1，如果新页面在缓存中，只是激活缓存，触发新页面的activated事件
     * 2，如果新页面没在缓存中，重新打开新页面，触发新页面的created事件
     * 3，新页面如果已经存在打开的多个，那么新页面重新创建一个新的，触发新页面的created事件
     * @param location 路由信息
     */
    Vue.prototype.$closePageAndActiveRouter = function (location) {
        this.$store.commit('removeTag', this.$vnode.key);
        this.$router.push(location);
    };
    /**
     * 关闭当前页面，清除当前页的keepalive缓存，并跳转到新页面，本方法将把与新页面路由相关的页面全部关闭，然后重新打开一个新页面，如：
     * 1,关闭新页面路由关联的所有已打开页面，并清空这些页面的缓存
     * 2，重新打开新页面，触发新页面的created事件
     * @param location 路由信息
     */
    Vue.prototype.$closePageAndRecreateRouter = function (location) {
        this.$store.commit('removeTag', this.$vnode.key);
        this.$store.commit('removeTagByName', location.name);
        this.$router.push(location);
    };

    Vue.prototype.$hasAuth = function (code) {
        let flag = true;
        if (typeof code === 'object' && Array.isArray(code)) {
            code.forEach(function (item, index) {
                if (!Util.showThisRoute(store.state.app.fontPermission, item)) {
                    flag = false;
                }
            });
            return flag;
        }
        return Util.showThisRoute(store.state.app.fontPermission, code);
    };
    Vue.prototype.$verifyPicUrl = function (url) {
        if (config.errortip) {
            return axios.defaults.baseURL + _.trim(url, '/');
        }
        return url;
    };
};

export default {
    install
};