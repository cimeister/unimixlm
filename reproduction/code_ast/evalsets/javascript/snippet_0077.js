import axios from 'axios';


const instance = axios.create({
    baseURL: 'https://smp-api.cloud.airscr.com',
});

instance.defaults.headers.common['token'] = 'AUTH';

export default instance;