/*
 * Co-Authored-By: dirigeants (https://github.com/dirigeants)
 * Co-Authored-By: Ravy <<EMAIL>> (https://ravy.pink)
 * License: MIT License
 * Credit example: Copyright (c) 2019 dirigeants, MIT License
 */
const { Argument, util: { regExpEsc } } = require('@aero/klasa');
const { Role } = require('discord.js');

const ROLE_REGEXP = Argument.regex.role;

function resolveRole(query, guild) {
	if (query instanceof Role) return guild.roles.cache.has(query.id) ? query : null;
	if (typeof query === 'string' && ROLE_REGEXP.test(query)) return guild.roles.cache.get(ROLE_REGEXP.exec(query)[1]);
	return null;
}

module.exports = class extends Argument {

	async run(arg, possible, msg) {
		if (!arg) throw 'No role provided.';
		if (!msg.guild) return this.store.get('role').run(arg, possible, msg);
		const resRole = resolveRole(arg, msg.guild);
		if (resRole) return resRole;

		const results = [];
		const reg = new RegExp(regExpEsc(arg), 'i');
		for (const role of msg.guild.roles.cache.values()) if (reg.test(role.name)) results.push(role);

		let querySearch;
		if (results.length > 0) {
			const regWord = new RegExp(`^${regExpEsc(arg)}$`, 'i');
			const filtered = results.filter(role => regWord.test(role.name));
			querySearch = filtered.length > 0 ? filtered : results;
		} else
			querySearch = results;


		if (querySearch.length) return querySearch[0];
		throw `\`${possible.name}\` must be a valid name, id or role mention`;
	}

};