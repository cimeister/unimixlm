<h1><?= $title ?></h1>

<?php if (is_array($tags)) : ?>
<div id="tags">
<?php foreach ($tags as $tag) : ?>
    <div class="tag small">
        <p class="tag-info">
            <a href='<?=$this->url->create('questions/tag-id/' . $tag->id)?>'><?= $tag->label ?></a>
            <span>x <?= $tag->numQuestions ?></span>
        </p>
        <p class="tag-content">
            <?= $tag->description ?>
        </p>
    </div>
<?php endforeach; ?>
</div>
<?php endif; ?>