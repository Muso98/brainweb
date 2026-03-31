from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tumor', '0004_airesult_bbox_diag_mm_airesult_report_generated_at_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='airesult',
            name='mask_nifti',
            field=models.FileField(upload_to='results/masks/', blank=True, null=True),
        ),
    ]
