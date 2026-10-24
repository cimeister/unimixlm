import Tweet from "../Models/Tweet";
import * as FileSystem from "fs-extra";
import TweetLabel from "../Models/LabeledTweet";
import ITweetRepository from "../Interfaces/ITweetRepository";
import ITwitterDataService from "../Interfaces/ITwitterDataService";
import {inject, injectable} from "inversify";
import Component from "../Infrastructure/Component";
import ITweetCrawlService from "../Interfaces/ITweetCrawlService";
import ITwitterUserRepository from "../Interfaces/ITwitterUserRepository";

@injectable()
export default class TweetCrawlService implements ITweetCrawlService {

    private count: number;
    private tweetRepo: ITweetRepository;
    private userRepo: ITwitterUserRepository;
    private twitterService: ITwitterDataService;

    public constructor(
        @inject(Component.TweetRepository) tweetRepo: ITweetRepository,
        @inject(Component.TwitterUserRepository) userRepo: ITwitterUserRepository,
        @inject(Component.TwitterDataService) twitterService: ITwitterDataService
    ) {
        this.tweetRepo = tweetRepo;
        this.twitterService = twitterService;
        this.userRepo = userRepo;
        this.count = 0;
    }

    public async getAll(): Promise<void> {
        const labeledTweets = await this.getLabeledTweets();
        const separatedTweets: TweetLabel[][] = this.splitJobs(labeledTweets, 25);
        this.processJobs(separatedTweets, async tweetLabels => {
            for (let tweetLabel of tweetLabels) {
                const id: string = tweetLabel.id;
                if (! await this.twitterExceptions(async () => {
                    const tweet = await this.twitterService.getTweet(id);
                    console.log(tweet.user_id);
                    await this.storeTweet(tweet, tweetLabel);
                })) return false;
                console.log("Processed Tweets: " + ++this.count);
            }
            return true;
        }, 1000 * 30);
    }

    public async storeUsers(): Promise<void> {
        const userIds = await this.tweetRepo.getAllUserId();
        console.log(userIds.length);
        const separatedIds: string[][] = this.splitJobs(userIds, 25);
        this.processJobs(separatedIds, async (values) => {
            for (let id of values) {
                if (! await this.twitterExceptions(async () => {
                    const user = await this.twitterService.getUser(id);
                    await this.userRepo.create(user);
                })) return false;
                console.log("Processed Users: " + ++this.count);
            }
            return true;
        }, 1000 * 30);
    }

    private async getLabeledTweets(): Promise<TweetLabel[]> {
        const dataset: Buffer = await FileSystem.readFile(__dirname + "/../../res/twitterDataset.csv");
        const rows: string[] = dataset.toString().split("\n");
        const labeledTweets: TweetLabel[] = [];
        for (let rawRow of rows) {
            const row: string[] = rawRow.split(",");
            if (row[1] == "racism") continue;
            labeledTweets.push({
                id: row[0],
                label: row[1]
            });
        }
        return labeledTweets;
    }

    private async storeTweet(tweet: Tweet, tweetLabel: TweetLabel): Promise<void> {
        tweet.label = tweetLabel.label;
        try {
            await this.tweetRepo.create(tweet);
        } catch(dbErr) {
            console.error(dbErr);
            console.log(tweet);
        }
    }

    private async processJobs<T>(values: T[][], action: (values: T[]) => Promise<boolean>, delay: number) {
        return new Promise<void>(async (res, rej) => {
            let i = 0;
            const jobSetCount = values.length;
            const id = setInterval(async() => {
                if (i == jobSetCount) {
                    clearInterval(id);
                    res();
                }
                if (await action(values[i])) i++;
            }, delay);
            if (await action(values[i])) i++;
        });
    }

    private splitJobs(values: any[], count: number): any[][] {
        const result: any[][] = [];
        let jobSetCounter = 0;
        while (jobSetCounter * count < values.length) {
            const jobSet = [];
            for (let i = 0; i < count; i++) {
                jobSet.push(values[jobSetCounter * count + i]);
            }
            result.push(jobSet);
            jobSetCounter++;
        }
        return result;
    }

    private async twitterExceptions(task: () => void): Promise<boolean> {
        try {
            await task();
        } catch (err) {
            const error = err as Error;
            console.log(error.message);
            if (error.message == "429") return false; // Twitter API is rejecting too many connections
        }
        return true;
    }

}