






function NumberToWords() {

    var units = [ " ", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten" ];
    var teens = [ "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty" ];
    var tens = [ "", "ten", "twenty", "thirty", "forty", "fifty", "sixty",
        "seventy", "eighty", "ninety" ];

    var othersBd = [ "thousand", "lac", "crore" ];

    var othersIntl = [ "thousand", "million", "billion", "trillion" ];

    var BD_MODE = "bd";
    var INTERNATIONAL_MODE = "international";
    var currentMode = BD_MODE;

    var getBelowHundred = function(n) {
        if (n >= 100) {
            return "greater than or equal to 100";
        };
        if (n <= 10) {
            return units[n];
        };
        if (n <= 20) {
            return teens[n - 10 - 1];
        };
        var unit = Math.floor(n % 10);
        n /= 10;
        var ten = Math.floor(n % 10);
        var tenWord = (ten > 0 ? (tens[ten] + " ") : '');
        var unitWord = (unit > 0 ? units[unit] : '');
        return tenWord + unitWord;
    };

    var getBelowThousand = function(n) {
        if (n >= 1000) {
            return "greater than or equal to 1000";
        };
        var word = getBelowHundred(Math.floor(n % 100));

        n = Math.floor(n / 100);
        var hun = Math.floor(n % 10);
        word = (hun > 0 ? (units[hun] + " hundred ") : '') + word;

        return word;
    };

    return {
        numberToWords : function(n) {
            if (isNaN(n)) {
                return "Not a number";
            };

            var word = '';
            var val;

            val = Math.floor(n % 1000);
            n = Math.floor(n / 1000);

            word = getBelowThousand(val);

            if (this.currentMode == BD_MODE) {
                othersArr = othersBd;
                divisor = 100;
                func = getBelowHundred;
            } else if (this.currentMode == INTERNATIONAL_MODE) {
                othersArr = othersIntl;
                divisor = 1000;
                func = getBelowThousand;
            } else {
                throw "Invalid mode - '" + this.currentMode
                + "'. Supported modes: " + BD_MODE + "|"
                + INTERNATIONAL_MODE;
            };

            var i = 0;
            while (n > 0) {
                if (i == othersArr.length - 1) {
                    word = this.numberToWords(n) + " " + othersArr[i] + " "
                    + word;
                    break;
                };
                val = Math.floor(n % divisor);
                n = Math.floor(n / divisor);
                if (val != 0) {
                    word = func(val) + " " + othersArr[i] + " " + word;
                };
                i++;
            };
            return word;
        },
        setMode : function(mode) {
            if (mode != BD_MODE && mode != INTERNATIONAL_MODE) {
                throw "Invalid mode specified - '" + mode
                + "'. Supported modes: " + BD_MODE + "|"
                + INTERNATIONAL_MODE;
            };
            this.currentMode = mode;
        }
    }
}