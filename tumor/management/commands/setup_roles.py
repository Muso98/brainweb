from django.core.management.base import BaseCommand
from django.contrib.auth.models import Group, Permission

class Command(BaseCommand):
    help = "Create default BrainWeb roles and assign permissions."

    def handle(self, *args, **kwargs):

        # Guruhlar
        groups = ["Clinician", "Radiologist", "Technician"]
        for name in groups:
            Group.objects.get_or_create(name=name)

        clinician = Group.objects.get(name="Clinician")
        radiologist = Group.objects.get(name="Radiologist")
        technician = Group.objects.get(name="Technician")

        # Permissionlarni olish
        perms = Permission.objects.filter(codename__in=[
            "upload_study",
            "review_study",
            "view_study",
            "add_study"
        ])

        # Clinician huquqlari
        clinician.permissions.set(perms.exclude(codename="review_study"))

        # Technician huquqlari
        technician.permissions.set(perms.exclude(codename="review_study"))

        # Radiologist huquqlari
        radiologist.permissions.set(perms)  # hammasi bor

        self.stdout.write(self.style.SUCCESS("Roles created and permissions assigned."))
