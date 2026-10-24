import json
import requests
from six import add_metaclass
from abc import ABCMeta, abstractmethod
from rauth import OAuth1Service, OAuth2Service
from flask import url_for, request, redirect


def decoder(payload):
  return json.loads(payload.decode('utf-8'))


@add_metaclass(ABCMeta)
class OAuthBase(object):
  def __init__(self, name, config):
    self.name = name
    self.client_id = config.client_id
    self.client_secret = config.secret_id

  @abstractmethod
  def authorize(self):
    """Start the authorization process."""

  @abstractmethod
  def callback(self):
    """Returns the data needed to authorize the user."""

  @property
  def redirect_uri(self):
    return url_for('auth.oauth_callback', provider=self.name,
        next = request.args.get('next') or request.referrer or None,
        _external=True
    )


class OAuthFactory(object):
  _providers = None

  @classmethod
  def get_provider(cls, provider, config=None):
    if cls._providers is None:
      cls._providers = {}
    if provider not in cls._providers:
      if provider == 'facebook':
        cls._providers[provider] = OAuthFacebook(config)
      else:
        cls._providers[provider] = OAuthGoogle(config)
    return cls._providers[provider]

  @classmethod
  def dispose(cls):
    providers = cls._providers
    for provider in providers:
      del provider
    del cls._providers
    cls._providers = None


class OAuthFacebook(OAuthBase):
  def __init__(self, credentials):
    super(OAuthFacebook, self).__init__('facebook', credentials)
    self.service = OAuth2Service(
        name='facebook',
        client_id=self.client_id,
        client_secret=self.client_secret,
        authorize_url='https://graph.facebook.com/oauth/authorize',
        access_token_url='https://graph.facebook.com/oauth/access_token',
        base_url='https://graph.facebook.com/'
    )

  def authorize(self):
    return redirect(self.service.get_authorize_url(
        scope='email',
        response_type='code',
        redirect_uri=self.redirect_uri)
    )

  def callback(self):
    if 'code' not in request.args:
      return None, None, None
    code = request.args.get('code')
    data = {
      'code': code,
      'redirect_uri': self.redirect_uri,
      'grant_type': 'authorization_code'
    }
    session = self.service.get_auth_session(data=data, decoder=decoder)
    me = session.get('me?fields=id,email').json()
    return ('facebook$' + me['id'],
            me.get('email').split('@')[0],
            me.get('email')
    )


class OAuthGoogle(OAuthBase):
  def __init__(self, credentials):
    super(OAuthGoogle, self).__init__('google', credentials)
    self.service = OAuth2Service(
        name='google',
        client_id=self.client_id,
        client_secret=self.client_secret,
        authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
        access_token_url='https://www.googleapis.com/oauth2/v4/token',
        base_url='https://www.googleapis.com/oauth2/v3/'
    )

  def authorize(self):
    return redirect(self.service.get_authorize_url(
        scope='email profile openid',
        response_type='code',
        access_type='offline',
        prompt='consent select_account',
        redirect_uri=self.redirect_uri)
    )

  @property
  def redirect_uri(self):
    return url_for('auth.oauth_callback', provider=self.name,
        _external=True
    )

  def fetch_user_info(self, info):
    email = info.get('email')
    social_id = info.get('sub')
    username = info.get('name')
    return social_id, username, email

  def callback(self):
    code = request.args.get('code')
    if not code:
      return None, None, None
    data = {
      'code': code,
      'redirect_uri': self.redirect_uri,
      'grant_type': 'authorization_code'
    }
    session = self.service.get_auth_session(
      data=data, key='id_token', decoder=decoder
    )
    info = session.get('tokeninfo',
        params=dict(id_token=session.access_token)).json()
    return self.fetch_user_info(info)