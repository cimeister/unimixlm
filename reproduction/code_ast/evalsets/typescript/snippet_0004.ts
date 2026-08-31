type ResponseBody = {
    url?: string;
    album?: AlbumRaw;
    artist?: ArtistRaw;
    lyrics?: LyricsRaw;
    richsync?: RichSyncRaw;
    snippet?: SnippetRaw;
    subtitle?: SubtitleRaw;
    subtitle_translated?: SubtitleRaw;
    track?: TrackRaw;

    album_list: { album: AlbumRaw }[];
    artist_list: { artist: ArtistRaw }[];
    track_list: { track: TrackRaw }[];
    music_genre_list: { music_genre: GenreRaw }[];
};

type ResponseRaw = {
    message: {
        header: {
            status_code: number;
            execute_time: number;
            available?: number;
            hint?: string;
        },
        body: {
            [T in keyof ResponseBody]: ResponseBody[T]
        } | []
    }
}

type IndicatorRaw = {
    instrumental?: number;
    explicit?: number;
    restricted?: number;
    managed?: number;
    has_subtitles?: number;
    has_lyrics?: number;
    has_richsync?: number;
    verified?: number;
    can_edit?: number;
    locked?: number;
}

type TrackingRaw = {
    script_tracking_url?: string;
    pixel_tracking_url?: string;
    html_tracking_url?: string;
}

type TrackRaw = {
    // Track
    track_name?: string;
    track_name_translation_list?: string[];
    track_rating?: number;
    track_length?: number;
    track_edit_url?: string;
    track_share_url?: string;
    first_release_date?: string;
    updated_time?: string;
    num_favourite?: number;

    // Artist
    artist_id?: number;
    artist_name?: string;
    artist_mbid?: string;

    // Album
    album_id?: number;
    album_name?: string;
    album_coverart_100x100?: string;
    album_coverart_350x350?: string;
    album_coverart_500x500?: string;
    album_coverart_800x800?: string;

    // Track related IDs
    track_id?: number;
    track_xboxmusic_id?: string;
    track_mbid?: string;
    track_spotify_id?: string;
    track_soundcloud_id?: number;
    track_isrc?: string;
    commontrack_id?: number;
    commontrack_vanity_id?: string;

    // Lyrics/subtitle
    subtitle_id?: number;
    lyrics_id?: number;

    // Genres
    primary_genres?: {
        music_genre_list?: ResponseBody["music_genre_list"];
    };
    secondary_genres?: {
        music_genre_list?: ResponseBody["music_genre_list"];
    };
} & IndicatorRaw;

type GenreRaw = {
    music_genre_vanity: string;
    music_genre_name_extended: string;
    music_genre_name: string;
    music_genre_parent_id: number;
    music_genre_id: number
}

type LyricsRaw = {
    lyrics_language_description?: string;
    lyrics_language?: string;
    lyrics_body?: string;
    lyrics_id?: number;

    writer_list?: string[];
    publisher_list?: string[];
    action_requested?: string;

    lyrics_copyright?: string;
    updated_time?: string;

    translated_lyrics?: {
        selected_language: string;
        lyrics_body: string;
    } & TrackingRaw;
} & TrackingRaw & IndicatorRaw;

type SnippetRaw = {
    snippet_language?: string;
    snippet_body?: string;
    snippet_id?: number;
    updated_time?: string;
} & TrackingRaw & IndicatorRaw;

type SubtitleRaw = {
    subtitle_language_description?: string;
    subtitle_language?: string;
    subtitle_body?: string;
    subtitle_id?: number;
    subtitle_length?: number;

    writer_list?: string[];
    publisher_list?: string[];
    lyrics_copyright?: string;
    updated_time?: string;
} & TrackingRaw & IndicatorRaw;

type RichSyncRaw = {
    richsync_language_description?: string;
    richsync_language?: string;
    richsync_body?: string;
    richsync_id?: number;
    richsync_length?: number;

    writer_list?: string[];
    publisher_list?: string[];
    lyrics_copyright?: string;
    updated_time?: string;
} & TrackingRaw & IndicatorRaw;

type ArtistRaw = {
    // Artist
    artist_name: string;
    artist_name_translation_list: {
        artist_name_translation: {
            language: string;
            translation: string;
        };
    }[];
    artist_rating: number;
    artist_edit_url: string;
    artist_share_url: string;
    artist_twitter_url: string;
    updated_time: string;
    artist_credits: {
        artist_list: ResponseBody["artist_list"]
    };
    artist_alias_list: {
        artist_alias: string;
    }[];
    artist_country: string;
    artist_comment: string;
    artist_vanity_id: string;

    // Artist related IDs
    artist_id: number;
    artist_mbid: string;

    // Genres
    primary_genres: {
        music_genre_list?: ResponseBody["music_genre_list"];
    };
    secondary_genres?: {
        music_genre_list?: ResponseBody["music_genre_list"];
    };
} & IndicatorRaw;

type AlbumRaw = {
    // Album
    album_name?: string;
    album_rating?: number;
    album_track_count?: number;
    album_edit_url?: string;
    album_share_url?: string;
    album_release_type?: string;
    album_release_date?: string;
    album_copyright?: string;
    updated_time?: string;

    album_pline?: string;
    album_vanity_id?: string;
    album_label?: string;
    album_coverart_100x100?: string;
    album_coverart_350x350?: string;
    album_coverart_500x500?: string;
    album_coverart_800x800?: string;

    // Artist
    artist_id?: number;
    artist_name?: string;
    artist_mbid?: string;

    // Album related IDs
    album_id?: number;
    album_mbid?: string;

    // Genres
    primary_genres?: {
        music_genre_list?: ResponseBody["music_genre_list"];
    };
    secondary_genres?: {
        music_genre_list?: ResponseBody["music_genre_list"];
    };
} & IndicatorRaw;

type Result<T> = {
    error: boolean;
    errorCode?: string | number;
    message?: string;
    data?: T;
}

type Acceptables = {
    [key: string]: {
        key: string;
        required?: boolean;
    }
}

type Trackings = {
    script?: string;
    pixel?: string;
    html?: string;
}

type CoverArt = {
    x100?: string;
    x350?: string;
    x500?: string;
    x800?: string;
}