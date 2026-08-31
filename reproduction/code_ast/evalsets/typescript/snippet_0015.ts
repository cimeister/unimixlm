/* eslint-disable */
import MockAdapter from 'axios-mock-adapter';

import {
  snakifyKeys,
  firstUp,
  isArray,
  isFn,
  stringify,
  warn
} from './utils';
import { mockDefaultConfig, httpMethodList, useDataMethodList } from './config';

/* declare */
import {
  MockAdapterOptions,
  IConfigHandlerInputs,
  IMockHandlerParams,
  ICreateAPIOptions,
  AxiosInstance,
  AxiosRequestConfig,
  IAxiosRequest,
  IUseObject
} from '../types/index.d';

// Cache for whole page
const RequestCache = new Set();

/**
 * Create config for real axios request
 * 
 * @param instance 
 * @param config 
 */
function configHandler({
  methodUp,
  beforeRequest,
  data
}: IConfigHandlerInputs) {
  const config = [];
  if (useDataMethodList.has(methodUp as any)) {
    config[0] = data;
    beforeRequest
      ? config[1] = beforeRequest({})
      : false;
  } else {
    config[0] = beforeRequest ? beforeRequest(data) : data;
  }
  return config;
}

/**
 * Create mock adapter instance
 * 
 * @param instance [required]: Axios instance
 * @param config [optional]: Mock instance options
 */
function createMock(
  instance: AxiosInstance,
  config: MockAdapterOptions = {},
) {
  return new MockAdapter(instance, (Object as any).assign(
    mockDefaultConfig,
    config,
  ));
}

/**
 * Define mock api with given mockData
 * 
 * @param this [required]: mock instance
 * @param param1 [required]: api params
 * @param mockData [required]: mock data list
 */
function mockHandler(
  this: MockAdapter,
  { method, svc, config }: IMockHandlerParams,
  mockReply: Function | any[],
  once: Boolean,
): void {
  const mock = this;
  // handler choose
  let handler: Function | any[];
  if (isFn(mockReply)) {
    handler = function mockReplyHandler(mockConfig: MockAdapterOptions) {
      return new Promise(
        (resolve: Function, reject: Function) => (mockReply as Function)(resolve, reject, mockConfig)
      );
    };
  } else {
    handler = mockReply;
  }
  // config handling
  (mock as any)
    [`on${firstUp(method)}`](svc, (config as any[])[0])
    [once ? 'replyOnce' : 'reply'].apply(mock, isFn(handler) ? [handler] : handler);
}

/**
 * Class of AxiosRequest
 * user can create & manipulate with this object
 * 
 * @param instance [required]: Axios intance
 * @param options [required]: shim configuration
 */
export function AxiosRequest(
  this: IAxiosRequest,
  instance: AxiosInstance,
  options: ICreateAPIOptions,
) {
  this.$options = options;
  this.$instance = instance;
  this.$adapter = null;
  this.ReplyCache = new Map();
  this.runBuilder = () => {};
  this.init();
}

AxiosRequest.prototype = {

  init(): void {
    const { $instance } = this;
    const { useMock } = this.$options;
    if (useMock) {
      this.$adapter = createMock($instance);
    }
    // runBuilder return the correct method for run
    this.runBuilder = useMock
      ? (...args: any) => () => this._useMockRequest(...args)
      : (...args: any) => () => this._normalRequest(...args);
  },

  use(method: String, svc: String, data: any = {}): IUseObject {
    const parent = this;
    const { _mock, ReplyCache, $options } = this;
    const { snakifyData, beforeRequest } = $options;
    const methodUp = method.toUpperCase();
    data = snakifyData ? snakifyKeys(data) : data;
    let configs = {
      method,
      svc,
      data,
      config: configHandler({ methodUp, beforeRequest, data }),
    };
    const cacheToken = stringify(configs);
    const useOnce = true;

    // Return an object to define mock data & calling
    return {
      with(fn: Function | any[]) {
        const { useMock } = parent.$options;
        if (!useMock) return this;
        const invalid = !isFn(fn) && !isArray(fn);
        if (invalid) return warn(
          'reply invalid, should be type Function or Array',
          fn,
        );
        ReplyCache.set(cacheToken, fn);
        return this;
      },
      mock() {
        const { useMock } = parent.$options;
        if (!useMock) return this; // if dont use mock, return this.
        _mock.call(parent, configs, !useOnce);
        return this;
      },
      mockOnce() {
        const { useMock } = parent.$options;
        if (!useMock) return this; // if dont use mock, return this.
        _mock.call(parent, configs, useOnce);
        return this;
      },
      // pass configs to runBuilder
      // "run" will become a function, eg. "() => this._useMockRequest()"
      run: parent.runBuilder.call(parent, configs, !useOnce),
      runOnce: parent.runBuilder.call(parent, configs, useOnce)
    };
  },

  _mock(configs: IMockHandlerParams, once: Boolean) {
    const { $adapter, ReplyCache, $options } = this;
    const { anyReply } = $options;
    const cacheToken = stringify(configs);
    const hasCache = RequestCache.has(cacheToken);
    // check cache, avoid duplicate mocked
    if (!hasCache) {
      !once ? RequestCache.add(cacheToken) : false;
      // check if cache has the mock data
      if (ReplyCache.has(cacheToken)) {
        mockHandler.call(
          $adapter,
          configs,
          ReplyCache.get(cacheToken),
          once,
        );
      } else {
        anyReply && mockHandler.call(
          $adapter,
          configs,
          anyReply,
          once,
        );
      }
    }
  },

  _useMockRequest(configs: IMockHandlerParams, once: Boolean): AxiosRequestConfig {
    const { _mock, _normalRequest } = this;

    // mock api
    _mock.call(this, configs, once);

    // Important!! Don't remove this return
    // This return stays for Promise mechanism
    return _normalRequest.call(this, configs);
  },

  _normalRequest({ method, svc, config }: IMockHandlerParams)
    : AxiosRequestConfig | void {
    const { $instance, $options } = this;
    const { beforeResponse } = $options;
    if (!httpMethodList.has(method.toUpperCase())) return warn(
      'Invalid http method',
      method,
    );
    return $instance[method.toLowerCase()](
      svc,
      ...(config as any[]),
    ).then(beforeResponse ? beforeResponse : (res: any) => res);
  },
}