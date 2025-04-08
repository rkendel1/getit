from django.db import models

class DocumentCategory(models.Model):
    name = models.CharField(max_length=100, unique=True)

    class Meta:
        verbose_name_plural = "Document Categories"
        ordering = ['name']

    def __str__(self):
        return self.name