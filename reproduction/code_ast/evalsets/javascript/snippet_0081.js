// 知识库片区库相关
const rentRouterMap = [
    {
        path: '/customers/rent',
        component: resolve => require( ['@/views/Customers/Rent'],resolve),
        name: 'customersRent',
        meta: {
            title: 'customersRent',
            // icon: 'users',
            roles: ['sysme']
        }
    },
    // {
    //   hidden: true,
    //   path: '/building/rentInfo/:id',
    //   component: resolve => require( ['./../../views/building/rent/rentInfo/rentInfo'],resolve),
    //   name: 'rentInfo',
    //   // meta: { title: 'rentInfo', icon: 'kinfo', roles: ['customers'] }
    //   meta: { title: 'rentInfo', roles: ['customers'] }
    // },
]

export default {
    name:'rent',
    route: rentRouterMap
}