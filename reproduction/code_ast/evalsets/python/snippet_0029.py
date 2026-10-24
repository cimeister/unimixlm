# -*- coding: utf-8 -*-

"""Monitoring library for Puppet Agent and Puppet Server components.

Implements a number of interfaces for interacting with Puppet Agent state and
provides the ability to reliably determine the status of a Puppet Agent,
profile catalog run performance and detect failed resources and events.

Also provides interfaces for interacting with Puppet Server service APIs to
reliably detect the status of critical services on Puppet.

Attributes:
    LOG (:class:`logging.Logger`): A module level logger instance. The
        configuration of handlers is the prerogative of developers consuming
        the library.
"""

import logging
import os

import yaml

from .utils import get_timedelta, safe_get

LOG = logging.getLogger('puppetbeacon.monitor')


class AgentState(object):
    """Base class that provides an interface to the Puppet agent state.

    Provides an interface that can be used to interact with the various Puppet
    agent state objects to determine if an agent is enabled or actively
    executing a catalog run. Also provides a method for retrieving
    detailed statistics from the last catalog run.

    Args:
        summary_file (:obj:`str`, optional): Fully-qualified path to the Puppet
            Agent last run summary file. This file contains YAML structured
            data and statistics about the agents last catalog run.

            Default: /opt/puppetlabs/puppet/cache/state/last_run_summary.yaml
        run_lock (:obj:`str`, optional): Fully-qualified path to the Puppet
            Agent run lock. This lock is present if the agent is actively
            executing a catalog run.

            Default: /opt/puppetlabs/puppet/cache/state/agent_catalog_run.lock
        disabled_lock (:obj:`str`, optional): Fully-qualified path to the
            Puppet Agent disabled lock. This lock is present if the agent has
            been administratively disabled and contains an optional message.

            Default: /opt/puppetlabs/puppet/cache/state/agent_disabled.lock

    Attributes:
        disabled_message (:obj:`str`): If the agent has been administratively
            disabled, the disabled_message will be provided here if available.
            Defaults to None.
    """

    def __init__(self, summary_file=None, run_lock=None, disabled_lock=None):
        self.summary_file = summary_file if summary_file is not None else \
            '/opt/puppetlabs/puppet/cache/state/last_run_summary.yaml'
        self.run_lock = run_lock if run_lock is not None else \
            '/opt/puppetlabs/puppet/cache/state/agent_catalog_run.lock'
        self.disabled_lock = disabled_lock if disabled_lock is not None else \
            '/opt/puppetlabs/puppet/cache/state/agent_disabled.lock'

        self._disabled = None
        self.disabled_message = None

    @property
    def disabled(self):
        """:obj:`bool`: Puppet Agent administrative status. True if the agent
        is disabled, False otherwise.
        """

        try:
            with open(self.disabled_lock, 'r') as disabled_lock:
                LOG.debug('Located agent disabled lock at %s, looking ' +
                          'for an administrative message.', self.disabled_lock)
                self.disabled_message = \
                    safe_get(yaml.safe_load(disabled_lock), 'disabled_message')
                LOG.warning('Puppet agent has been administratively ' +
                            'disabled. Message: %s', self.disabled_message)
                return True
        except EnvironmentError:
            LOG.debug('No agent disabled lock present at %s, agent is ' +
                      'enabled.', self.disabled_lock)

        return False

    def get_run_summary(self):
        """Retrieves and deserializes agent run summary data and statistics.

        Provides an interface to interact with the last run summary data and
        statistics of a Puppet agent. Method performs a safe_load when
        deserializing to provide protection against execution of arbitrary
        code.

        Returns:
            dict: Returns a nested dictionary object with the deserialized data
            and statistics from the agent's last run. Returns an empty
            dictionary if the method is unable to retrieve or deserialize the
            summary data.
        """
        run_summary = None

        try:
            with open(self.summary_file, 'r') as summary_file:
                run_summary = yaml.safe_load(summary_file)
                LOG.debug('Successfully parsed Puppet agent summary data in ' +
                          'file %s', self.summary_file)
        except IOError as error:
            # TODO(e.westfall): Raise exception rather than return empty dict?
            LOG.error('Unable to locate or open summary file %s. Error: %s',
                      self.summary_file, error)
        except yaml.YAMLError as error:
            LOG.error('Unable to parse summary file %s. Error: %s',
                      self.summary_file, error)

        return run_summary if run_summary else {}


class PuppetAgent(AgentState):
    """Provides an interface to detailed Puppet agent data and statistics.

    Implements an interface that exposes data and statistics providing detailed
    information about resource and event failures, the last time the agent
    completed a catalog run and its duration, and the agent version.

    Also provides a method to determine if the agent is currently executing a
    catalog run and if so, its duration.

    Args:
        *args: Variable length argument list that is passed through to the
            :class:`AgentState` base class initializer.

            Can be used to override the default values for `summary_file`,
            `run_lock` and `disabled_lock` when directly instantiating a
            :class:`PuppetAgent` instance.
        **kwargs: Arbitrary keyword arguments that are passed through to the
            :class:`AgentState` base class.

            Can be used to override the default values for `summary_file`,
            `run_lock` and `disabled_lock` when directly instantiating a
            :class:`PuppetAgent` instance.

    Attributes:
        last_run (:obj:`int`): The number of seconds since the last catalog
            run.
        last_run_duration (:obj:`int`): The duration of the last catalog run
            in seconds.
        events_failed (:obj:`int`): The number of failed events during the last
            catalog run.
        resources_failed (:obj:`int`): The number of resources that failed
            during the last catalog run.
        resources_failed_restart (:obj:`int`): The number of resources that
            failed to restart during the last catalog run.
        puppet_version (:obj:`str`): The Puppet agent version.
    """

    def __init__(self, *args, **kwargs):
        AgentState.__init__(self, *args, **kwargs)

        self._run_duration = None
        self.last_run = None
        self.last_run_duration = None
        self.events_failed = None
        self.resources_failed = None
        self.resources_failed_restart = None
        self.puppet_version = None

        self.get_last_run()

    @property
    def run_duration(self):
        """Determine if a catalog run is in progress and return duration.

        Checks for the presence of the agent run lock and if present,
        determines the duration of the catalog run by calculating the age of
        the lock in seconds.

        Returns:
            int: An integer representing the number of seconds the run lock has
            been held by the Puppet agent.
        """

        run_duration = None

        try:
            run_duration = get_timedelta(os.path.getmtime(self.run_lock))
            LOG.debug('Located agent run lock at %s, Puppet agent has been ' +
                      'executing a catalog run for %s seconds.',
                      self.run_lock, run_duration)
        except OSError:
            LOG.debug('Puppet agent is not executing a catalog run.')

        return run_duration

    def get_last_run(self):
        """Obtains and processes summary data for last catalog run.

        Calls the :method:`get_run_summary` method from the :class:`AgentState`
        class to obtain last run summary data and statistics.

        Processes key data using the :func:`safe_get` helper function to safely
        evaluate nested values from the deserialized object. If values can be
        found, instance attributes are set.

        Returns:
            bool: Returns True if last run summary data was successfully
            retrieved and processed, returns False otherwise.
        """

        run_summary = self.get_run_summary()

        if not run_summary:
            return False

        self.last_run = \
            get_timedelta(safe_get(run_summary, 'time', 'last_run'))
        self.last_run_duration = \
            int(safe_get(run_summary, 'time', 'total'))
        self.events_failed = safe_get(run_summary, 'events', 'failure')
        self.resources_failed = \
            safe_get(run_summary, 'resources', 'failed')
        self.resources_failed_restart = \
            safe_get(run_summary, 'resources', 'failed_to_restart')
        self.puppet_version = safe_get(run_summary, 'version', 'puppet')

        return True