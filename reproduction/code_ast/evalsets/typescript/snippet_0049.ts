import {
	GuildMember,
	Message,
	MessageEmbed,
	MessageOptions,
	Snowflake,
	TextChannel,
} from 'discord.js'
import {
	ADMIN_CHANNELS,
	ADMIN_LOG_CHANNEL_ID,
	BACKUP_CHANNELS,
	ChannelsFilterType,
	IGNORED_CHANNELS,
	LOGGING_CHANNELS,
	MOD_LOG_CHANNEL_ID,
	PUBLIC_LOG_CHANNEL_ID,
} from '../config/channels.js'
import filterMessageByChannel from '../filterMessageByChannel.js'
import getPrimaryClient from '../getPrimaryClient.js'
import getMessageLink from './getMessageLink.js'
import getRelativeTime from './getRelativeTime.js'

enum EventName {
	MessageCreate = 'message',
	MessageDelete = 'messageDelete',
	MessageUpdate = 'messageUpdate',
	GuildMemberAdd = 'guildMemberAdd',
	GuildMemberRemove = 'guildMemberRemove',
}

const EVENT_EMBED_PROPERTIES = {
	[EventName.MessageCreate]: {
		color: 'GREEN',
		name: '🎆Message Created!🎆',
	},
	[EventName.MessageUpdate]: {
		color: 'YELLOW',
		name: '👷‍♀️Message Update👷‍♀️',
	},
	[EventName.MessageDelete]: {
		color: 'RED',
		name: '💢Message Delete💢',
	},
	[EventName.GuildMemberAdd]: {
		color: 'GREEN',
		name: '🎆Guild Member Add🎆',
	},
	[EventName.GuildMemberRemove]: {
		color: 'RED',
		name: '💢Guild Member Remove💢',
	},
}

async function sendLogForGuildMember(
	member: GuildMember,
	eventName: EventName,
	loggingChannelId: string,
): Promise<Message | null> {
	const loggingChannel = (await member.guild.channels.cache
		.get(loggingChannelId)
		.fetch()) as TextChannel
	const embedProperties = EVENT_EMBED_PROPERTIES[eventName]
	const messageEmbed = new MessageEmbed()
		.addFields(
			{
				name: 'Event Type',
				value: embedProperties.name,
			},
			{
				name: 'Member',
				value: `<@${member.user.id}>`,
			},
			{
				name: 'Member Created',
				value: getRelativeTime(new Date(member.user.createdTimestamp)),
			},
			{
				name: 'Member Created ISO 8601 Timestamp',
				value: new Date(member.user.createdTimestamp).toISOString(),
			},
		)
		.setColor(embedProperties.color)

	const messageOptions: MessageOptions = {
		embed: messageEmbed,
		disableMentions: 'all',
	}

	return await loggingChannel.send('', messageOptions)
}

async function setupGuildMemberDeleteEventHandler(
	loggingChannelId: Snowflake,
): Promise<void> {
	const DiscordClient = await getPrimaryClient()

	DiscordClient.on(
		EventName.GuildMemberRemove.toString(),
		async (member: GuildMember) => {
			await sendLogForGuildMember(
				member,
				EventName.GuildMemberRemove,
				loggingChannelId,
			)
		},
	)
}

async function setupGuildMemberAddEventHandler(
	loggingChannelId: Snowflake,
): Promise<void> {
	const DiscordClient = await getPrimaryClient()

	DiscordClient.on(
		EventName.GuildMemberAdd.toString(),
		async (member: GuildMember) => {
			await sendLogForGuildMember(
				member,
				EventName.GuildMemberAdd,
				loggingChannelId,
			)
		},
	)
}

async function setupMemberHandlers(loggingChannelId: Snowflake) {
	await setupGuildMemberDeleteEventHandler(loggingChannelId)
	await setupGuildMemberAddEventHandler(loggingChannelId)
}

// Return true if the event should not be logged, false if the event should be logged
function filterMessage(oldMessage: Message, newMessage: Message): boolean {
	// The only type of update that matters from a moderation perspective is content changes.
	// Changes in pin status do not matter.
	if (oldMessage && oldMessage.content === newMessage.content) {
		return true
	}

	// Do not log system pin notification additions or removals
	if (newMessage.type === 'PINS_ADD') {
		return true
	}

	return false
}

async function sendLogForMessage(
	message: Message,
	eventName: EventName,
	loggingChannelId: string,
	sendAttachments: boolean,
): Promise<Message | null> {
	const loggingChannel = (await message.guild.channels.cache
		.get(loggingChannelId)
		.fetch()) as TextChannel

	const embedProperties = EVENT_EMBED_PROPERTIES[eventName]
	const messageEmbed = new MessageEmbed()
		.addFields(
			{
				name: 'Event Type',
				value: embedProperties.name,
			},
			{
				name: 'Member',
				value: `<@${message.author.id}>`,
			},
			{
				name: 'Channel Name',
				value: `<#${message.channel.id}>`,
			},
			{
				name: 'Message Link',
				value: getMessageLink(message),
			},
		)
		.setColor(embedProperties.color)

	const messageOptions: MessageOptions = {
		embed: messageEmbed,
		disableMentions: 'all',
	}

	if (sendAttachments) {
		messageOptions['files'] = message.attachments.map((attachment) => ({
			attachment: attachment.url,
			name: attachment.name,
		}))
	}

	// TODO: for level 3 servers the limit of the bot should be the same as max possible (100mb)
	// however sometimes it failed locally on my machine... but on digital ocean it seems to succeed
	//
	// in other cases and in that failure case I could try uploading to s3 instead? then link it in the message
	// and report in an error logs. use getSignedS3Url
	try {
		return await loggingChannel.send(message.content, messageOptions)
	} catch (error) {
		if (sendAttachments) {
			messageOptions['files'] = []
			messageEmbed.addField(
				'Attachment',
				'Failed to upload, report to developer',
			)

			return await loggingChannel.send(message.content, messageOptions)
		}
		// TODO: figure out what to do in the case of a message send failing not for too large of an upload? is this even possible?
	}

	// TODO: figure out what to do in this case
	return Promise.resolve(null)
}

// TODO: consider adding a left/banned/kicked field (it's in audit log but at a glance is useful)
async function setupMessageDeleteHandler(
	loggingChannelId: Snowflake,
	channels: Snowflake[],
	channelsFilterType: ChannelsFilterType,
): Promise<void> {
	const DiscordClient = await getPrimaryClient()

	DiscordClient.on(
		EventName.MessageDelete,
		async (message: Message): Promise<void> => {
			if (
				filterMessageByChannel(message.channel.id, channels, channelsFilterType)
			) {
				return
			}

			if (filterMessage(null, message)) {
				return
			}

			// NOTE: attachments are not added because they will be removed after the message is deleted
			await sendLogForMessage(
				message,
				EventName.MessageDelete,
				loggingChannelId,
				false,
			)
		},
	)
}

async function setupMessageCreateHandler(
	loggingChannelId: Snowflake,
	channels: Snowflake[],
	channelsFilterType: ChannelsFilterType,
): Promise<void> {
	const DiscordClient = await getPrimaryClient()

	DiscordClient.on(
		EventName.MessageCreate,
		async (message: Message): Promise<void> => {
			if (
				filterMessageByChannel(message.channel.id, channels, channelsFilterType)
			) {
				return
			}

			if (filterMessage(null, message)) {
				return
			}

			await sendLogForMessage(
				message,
				EventName.MessageCreate,
				loggingChannelId,
				true,
			)
		},
	)
}

async function setupMessageUpdateHandler(
	loggingChannelId: Snowflake,
	channels: Snowflake[],
	channelsFilterType: ChannelsFilterType,
): Promise<void> {
	const DiscordClient = await getPrimaryClient()

	DiscordClient.on(
		EventName.MessageUpdate,
		async (oldMessage: Message, newMessage: Message): Promise<void> => {
			if (
				filterMessageByChannel(
					newMessage.channel.id,
					channels,
					channelsFilterType,
				)
			) {
				return
			}

			if (filterMessage(oldMessage, newMessage)) {
				return
			}

			// NOTE: attachments are not added because they cannot be updated after the message is created and the message created
			// log includes the attachments
			await sendLogForMessage(
				newMessage,
				EventName.MessageUpdate,
				loggingChannelId,
				false,
			)
		},
	)
}

// TODO: figure out a way for these to actually show the emojis rather than text
async function setupMessageHandlers(
	loggingChannelId: Snowflake,
	channels: Snowflake[],
	channelsFilterType: ChannelsFilterType,
): Promise<void> {
	await setupMessageCreateHandler(
		loggingChannelId,
		channels,
		channelsFilterType,
	)
	await setupMessageDeleteHandler(
		loggingChannelId,
		channels,
		channelsFilterType,
	)
	await setupMessageUpdateHandler(
		loggingChannelId,
		channels,
		channelsFilterType,
	)
}

// TODO: Create a fallback for too much text? need to test it out
// TODO: maybe reduce some duplication between each message and guild member handler
// TODO: add category whitelist/blacklist??
export default async function setupLogging(): Promise<void> {
	setupMemberHandlers(PUBLIC_LOG_CHANNEL_ID)
	await setupMessageHandlers(
		MOD_LOG_CHANNEL_ID,
		LOGGING_CHANNELS.concat(ADMIN_CHANNELS, IGNORED_CHANNELS, BACKUP_CHANNELS),
		ChannelsFilterType.Deny,
	)
	await setupMessageHandlers(
		ADMIN_LOG_CHANNEL_ID,
		ADMIN_CHANNELS,
		ChannelsFilterType.Allow,
	)
}