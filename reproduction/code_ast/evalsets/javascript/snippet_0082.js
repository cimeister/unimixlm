const { createOrUpdateData } = require('../utils/functions')

module.exports = {
    async generateDbWithStarshipsNames(pilotsComImg, starships) {
        await starships.map((starship) => {
            pilotsComImg.map((pilot) => {
                if (pilot.starships) {
                    pilot.starships.map((starshipitem, i) => {
                        if (starshipitem === starship.url) {
                            pilot.starships[i] = { ...starship }


                        }
                    })
                }
            })
        })

        return await createOrUpdateData('pilotsComImgStarshipName.json', pilotsComImg)
    }
}