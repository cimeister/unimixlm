export default class KickerGamesList extends HTMLElement {
    constructor() {
        super();
        this.innerHTML = `
        <table>
        <thead>
            <tr>
                <th>date</th>
                <th>team 1</th>
                <th>elo</th>
                <th>score</th>
                <th>elo</th>
                <th>team 2</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td colspan="6">Loading...</td>
            </tr>
        </tbody>
        </table>
        `;
    }

    connectedCallback() {
        fetch("/games")
            .then(response => response.json())
            .then(games => this.renderGames(games));
    }

    renderGames(games){
        this.querySelector("tbody").innerHTML = games.map(game =>
            `
            <tr>
                <td>${game.date}</td>
                <td>${game.team1Player1} & ${game.team1Player2}</td>
                <td>${game.eloTeam1}</td>
                <td>${game.scoreGame1} ${game.scoreGame2}</td>
                <td>${game.eloTeam2}</td>
                <td>${game.team2Player1} & ${game.team2Player2}</td>
            </tr>
            `
        ).reduce((acc, cur) => acc + cur, "");
    }
}